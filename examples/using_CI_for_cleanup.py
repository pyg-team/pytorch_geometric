import argparse
import copy
import math
import os
from typing import Dict, List

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from inferred_stypes import dataset2inferred_stypes
from model import Model
from text_embedder import GloveTextEmbedding
from torch.nn import BCEWithLogitsLoss, L1Loss
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
from torch_geometric.seed import seed_everything
from tqdm import tqdm


from relbench.data import NodeTask, RelBenchDataset
from relbench.data.task_base import TaskType
from relbench.datasets import get_dataset
from relbench.external.graph import get_node_train_table_input, make_pkey_fkey_graph


def init_pytorch_worker(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)


def run(rank, args, data, col_to_stype_dict, task, world_size)
    if world_size == 0:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:" + str(rank))
    init_pytorch_worker(
        rank,
        world_size,
    )
    data, col_stats_dict = make_pkey_fkey_graph(
        dataset.db,
        col_to_stype_dict=col_to_stype_dict,
        text_embedder_cfg=TextEmbedderConfig(
            text_embedder=GloveTextEmbedding(device=device), batch_size=256
        ),
        cache_dir=os.path.join(root_dir, f"{args.dataset}_materialized_cache"),
    )
    loader_dict: Dict[str, NeighborLoader] = {}
    for split, table in [
        ("train", task.train_table),
        ("val", task.val_table),
        ("test", task.test_table),
    ]:
        table_input = get_node_train_table_input(table=table, task=task)
        entity_table = table_input.nodes[0]
        if world_size > 1:
            idx_for_this_rank = table_input.split(
                table_input.size(0) // world_size, dim=0)[rank].clone()
        loader_dict[split] = NeighborLoader(
            data,
            num_neighbors=[
                int(args.num_neighbors / 2**i) for i in range(args.num_layers)
            ],
            time_attr="time",
            input_nodes=idxs_for_this_rank,
            input_time=table_input.time,
            transform=table_input.transform,
            batch_size=args.batch_size,
            temporal_strategy=args.temporal_strategy,
            shuffle=split == "train",
            num_workers=args.num_workers,
            persistent_workers=args.num_workers > 0,
        )

    clamp_min, clamp_max = None, None
    if task.task_type == TaskType.BINARY_CLASSIFICATION:
        out_channels = 1
        loss_fn = BCEWithLogitsLoss()
        tune_metric = "roc_auc"
        higher_is_better = True
    elif task.task_type == TaskType.REGRESSION:
        out_channels = 1
        loss_fn = L1Loss()
        tune_metric = "mae"
        higher_is_better = False
        # Get the clamp value at inference time
        clamp_min, clamp_max = np.percentile(
            task.train_table.df[task.target_col].to_numpy(), [2, 98]
        )

    model = Model(
        data=data,
        col_stats_dict=col_stats_dict,
        num_layers=args.num_layers,
        channels=args.channels,
        out_channels=out_channels,
        aggr=args.aggr,
        norm="batch_norm",
    ).to(device)
    model = DistributedDataParallel(model, device_ids=[rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


    def train() -> float:
        model.train()

        loss_accum = count_accum = 0
        for batch in tqdm(loader_dict["train"]):
            batch = batch.to(device)

            optimizer.zero_grad()
            pred = model(
                batch,
                task.entity_table,
            )
            pred = pred.view(-1) if pred.size(1) == 1 else pred
            loss = loss_fn(pred, batch[entity_table].y)
            loss.backward()
            optimizer.step()

            loss_accum += loss.detach().item() * pred.size(0)
            count_accum += pred.size(0)

        return loss_accum / count_accum


    @torch.no_grad()
    def test(loader: NeighborLoader) -> np.ndarray:
        model.eval()

        pred_list = []
        for batch in tqdm(loader):
            batch = batch.to(device)
            pred = model(
                batch,
                task.entity_table,
            )
            if task.task_type == TaskType.REGRESSION:
                assert clamp_min is not None
                assert clamp_max is not None
                pred = torch.clamp(pred, clamp_min, clamp_max)

            if task.task_type == TaskType.BINARY_CLASSIFICATION:
                pred = torch.sigmoid(pred)

            pred = pred.view(-1) if pred.size(1) == 1 else pred
            pred_list.append(pred.detach().cpu())
        return torch.cat(pred_list, dim=0).numpy()


    state_dict = None
    best_val_metric = 0 if higher_is_better else math.inf
    for epoch in range(1, args.epochs + 1):
        train_loss = train()
        val_pred = test(loader_dict["val"])
        val_metrics = task.evaluate(val_pred, task.val_table)
        if rank == 0:
            print(f"Epoch: {epoch:02d}, Train loss: {train_loss}, Val metrics: {val_metrics}")

        if (higher_is_better and val_metrics[tune_metric] > best_val_metric) or (
            not higher_is_better and val_metrics[tune_metric] < best_val_metric
        ):
            best_val_metric = val_metrics[tune_metric]
            state_dict = copy.deepcopy(model.state_dict())

    model.load_state_dict(state_dict)
    val_pred = test(loader_dict["val"])
    val_metrics = task.evaluate(val_pred, task.val_table)
    if rank == 0:
        print(f"Best Val metrics: {val_metrics}")

    test_pred = test(loader_dict["test"])
    test_metrics = task.evaluate(test_pred)
    for key, val in test_metrics.items():
        try:
            reduced_val = torch.tensor(val, dtype=torch.float, device=device)
            dist.all_reduce(reduce_val)
            test_metrics[key] = reduced_val
        except: # noqa
            pass

    if rank == 0:
        print(f"Best test metrics: {test_metrics}")
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="rel-stackex")
    parser.add_argument("--task", type=str, default="rel-stackex-engage")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--aggr", type=str, default="sum")
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_neighbors", type=int, default=128)
    parser.add_argument("--temporal_strategy", type=str, default="uniform")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument(
        "--n_devices", type=int, default=-1,
        help="1-8 to use that many GPUs. Defaults to all available GPUs")
    args = parser.parse_args()

    seed_everything(42)

    root_dir = "./data"

    # TODO: remove process=True once correct data/task is uploaded.
    dataset: RelBenchDataset = get_dataset(name=args.dataset, process=True)
    task: NodeTask = dataset.get_task(args.task, process=True)

    col_to_stype_dict = dataset2inferred_stypes[args.dataset]

    assert n_devices >= -1
    if torch.cuda.is_available():
        if args.n_devices == -1:
            world_size = torch.cuda.device_count()
        else:
            world_size = args.n_devices
    else:
        print("CUDA unavailable, using CPU...")
        world_size = 0
    print('Let\'s use', world_size, 'GPUs!')
    with tempfile.TemporaryDirectory() as tempdir:
        if world_size > 1:
            mp.spawn(
                run_train,
                args=(data, args, data, col_stats_dict, task, world_size),
                nprocs=world_size, join=True)
        else:
            run_train(0, data, args, data, col_to_stype_dict, task, world_size)
