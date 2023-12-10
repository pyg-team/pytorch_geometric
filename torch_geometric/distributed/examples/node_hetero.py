from torch_geometric.distributed.dist_context import DistContext
from torch_geometric.distributed.partition import load_partition_info

import argparse
import os.path as osp
import time

import torch
import torch.distributed
import torch.nn.functional as F

from ogb.nodeproppred import Evaluator
from torch.nn.parallel import DistributedDataParallel
from benchmark.utils.hetero_sage import HeteroGraphSAGE
from torch_geometric.nn import GraphSAGE, to_hetero

from torch_geometric.distributed import (
    LocalFeatureStore,
    LocalGraphStore,
    DistNeighborLoader,
)
from torch_geometric.distributed.rpc import rpc_barrier


import logging

logging.basicConfig(
    format='%(levelname)s:%(process)d:%(message)s', level=logging.INFO
)


@torch.no_grad()
def test(model, test_loader):
    evaluator = Evaluator(name="ogbn-mag")
    model.eval()
    xs = []
    y_true = []
    for i, batch in enumerate(test_loader):
        batch_size = batch["paper"].batch_size
        x = model(batch.x_dict, batch.edge_index_dict)
        x = x['paper'][:batch_size]
        xs.append(x.cpu())
        y_true.append(batch["paper"].y[:batch_size].cpu())
        print(f"---- test():  i={i}, batch={batch} ----")
        del batch
    y_pred = torch.cat(xs, dim=0).argmax(dim=-1, keepdim=True)
    y_true = torch.cat(y_true, dim=0).unsqueeze(-1)
    test_acc = evaluator.eval(
        {
            "y_true": y_true,
            "y_pred": y_pred,
        }
    )["acc"]
    return test_acc


def run_training_proc(
    local_proc_rank: int,
    num_nodes: int,
    node_rank: int,
    num_training_procs_per_node: int,
    dataset_name: str,
    root_dir: str,
    node_label_file: str,
    in_channels: int,
    out_channels: int,
    train_idx: torch.Tensor,
    test_idx: torch.Tensor,
    epochs: int,
    batch_size: int,
    master_addr: str,
    training_pg_master_port: int,
    train_loader_master_port: int,
    test_loader_master_port: int,
):
    graph = LocalGraphStore.from_partition(
        osp.join(root_dir, f"{dataset_name}-partitions"), node_rank
    )
    feature = LocalFeatureStore.from_partition(
        osp.join(root_dir, f"{dataset_name}-partitions"), node_rank
    )
    (
        meta,
        num_partitions,
        partition_idx,
        node_pb,
        edge_pb,
    ) = load_partition_info(
        osp.join(root_dir, f"{dataset_name}-partitions"), node_rank
    )
    print(f"-------- meta={meta}, partition_idx={partition_idx}")

    # node_pb = torch.cat(list(node_pb.values()))
    # edge_pb = torch.cat(list(edge_pb.values()))

    graph.num_partitions = num_partitions
    graph.partition_idx = partition_idx
    graph.node_pb = node_pb
    graph.edge_pb = edge_pb
    graph.meta = meta

    feature.num_partitions = num_partitions
    feature.partition_idx = partition_idx
    feature.meta = meta
    feature.node_feat_pb = node_pb
    feature.edge_feat_pb = edge_pb
    feature.labels = torch.load(node_label_file)

    partition_data = (feature, graph)

    # Initialize graphlearn_torch distributed worker group context.
    current_ctx = DistContext(
        world_size=num_nodes * num_training_procs_per_node,
        rank=node_rank * num_training_procs_per_node + local_proc_rank,
        global_world_size=num_nodes * num_training_procs_per_node,
        global_rank=node_rank * num_training_procs_per_node + local_proc_rank,
        group_name="distributed-sage-supervised-trainer",
    )

    # Initialize training process group of PyTorch.
    torch.distributed.init_process_group(
        backend="gloo",
        rank=current_ctx.rank,
        world_size=current_ctx.world_size,
        init_method="tcp://{}:{}".format(master_addr, training_pg_master_port),
    )

    # Basic params
    current_device = torch.device("cpu")
    num_workers = 2
    concurrency = 10
    batch_size = 1024
    num_layers = 3
    num_classes = 349
    num_neighbors = [15, 10, 5]
    async_sampling = False  # Hotfix: due to init_params() and loader RPC connection failures Hetero can only be processed in sync manner now

    # Create distributed neighbor loader for training
    train_idx = (
        "paper",
        train_idx.split(train_idx.size(0) // num_training_procs_per_node)[
            local_proc_rank
        ],
    )

    train_loader = DistNeighborLoader(
        data=partition_data,
        num_neighbors=num_neighbors,
        input_nodes=train_idx,
        batch_size=batch_size,
        shuffle=True,
        device=current_device,
        num_workers=num_workers,
        concurrency=concurrency,
        master_addr=master_addr,
        master_port=train_loader_master_port,
        async_sampling=async_sampling,
        current_ctx=current_ctx,
        disjoint=False,
    )

    @torch.no_grad()
    def init_params():
        # Initialize lazy parameters via forwarding a single batch to the model:
        batch = next(iter(train_loader))
        batch = batch.to(torch.device("cpu"))
        model(batch.x_dict, batch.edge_index_dict)

        del batch

    # Create distributed neighbor loader for testing.
    test_idx = (
        "paper",
        test_idx.split(test_idx.size(0) // num_training_procs_per_node)[
            local_proc_rank
        ],
    )
    test_loader = DistNeighborLoader(
        data=partition_data,
        num_neighbors=[-1],
        input_nodes=train_idx,
        batch_size=batch_size,
        shuffle=False,
        device=current_device,
        num_workers=num_workers,
        concurrency=concurrency,
        master_addr=master_addr,
        master_port=train_loader_master_port,
        async_sampling=async_sampling,
        current_ctx=current_ctx,
        disjoint=False,
    )
    # Define model and optimizer.
    # node_types = meta['node_types']
    # edge_types = [tuple(e) for e in meta['edge_types']]
    node_types = ["paper", "author"]
    edge_types = [
        ("paper", "cites", "paper"),
        ("paper", "rev_writes", "author"),
        ("author", "writes", "paper"),
    ]
    metadata = (node_types, edge_types)

    model = GraphSAGE(
        in_channels=128,
        hidden_channels=256,
        num_layers=num_layers,
        out_channels=num_classes,
    ).to(current_device)

    model = to_hetero(model, metadata)
    print(f"----------- init_params() ------------- ")
    init_params()
    torch.distributed.barrier()

    model = DistributedDataParallel(model, find_unused_parameters=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    print(f"----------- train() ------------- ")
    # Train and test.
    f = open("dist_sage_sup.txt", "a+")
    for epoch in range(0, epochs):
        model.train()
        start = time.time()
        for i, batch in enumerate(train_loader):
            batch_time = time.time()
            optimizer.zero_grad()
            out = model(batch.x_dict, batch.edge_index_dict)
            batch_size = batch["paper"].batch_size
            out = out['paper'][:batch_size]
            target = batch["paper"].y[:batch_size]
            loss = F.cross_entropy(out, target)
            loss.backward()
            optimizer.step()
            print(
                f"x2_worker: batch={batch}, cnt={i}, loss={loss}, time={time.time() - batch_time}"
            )
        end = time.time()
        f.write(
            f"-- [Trainer {current_ctx.rank}] Epoch: {epoch:03d}, Loss: {loss:.4f}, Epoch Time: {end - start}\n"
        )
        print(
            f"-- [Trainer {current_ctx.rank}] Epoch: {epoch:03d}, Loss: {loss:.4f}, Epoch Time: {end - start}\n"
        )
        # Test accuracy.
        if epoch % 5 == 0:
            print(f"----------- test() ------------- ")
            test_acc = test(model, test_loader)
            f.write(
                f"-- [Trainer {current_ctx.rank}] Test Accuracy: {test_acc:.4f}\n"
            )
            print(
                f"-- [Trainer {current_ctx.rank}] Test Accuracy: {test_acc:.4f}\n"
            )
    torch.distributed.barrier()

    print(f"----------- 555 ------------- ")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Arguments for distributed training of supervised SAGE."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ogbn-mag",
        help="The name of ogbn dataset.",
    )
    parser.add_argument(
        "--in_channel",
        type=int,
        default=128,
        help="in channel of the dataset, default is for ogbn-products",
    )
    parser.add_argument(
        "--out_channel",
        type=int,
        default=47,
        help="out channel of the dataset, default is for ogbn-products",
    )
    parser.add_argument(
        "--dataset_root_dir",
        type=str,
        default="../../data/mags",
        help="The root directory (relative path) of partitioned ogbn dataset.",
    )
    parser.add_argument(
        "--num_dataset_partitions",
        type=int,
        default=2,
        help="The number of partitions of ogbn-products dataset.",
    )
    parser.add_argument(
        "--num_nodes",
        type=int,
        default=2,
        help="Number of distributed nodes.",
    )
    parser.add_argument(
        "--node_rank",
        type=int,
        default=0,
        help="The current node rank.",
    )
    parser.add_argument(
        "--num_training_procs",
        type=int,
        default=2,
        help="The number of traning processes per node.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="The number of training epochs.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1024,
        help="Batch size for the training and testing dataloader.",
    )
    parser.add_argument(
        "--master_addr",
        type=str,
        default="localhost",
        help="The master address for RPC initialization.",
    )
    parser.add_argument(
        "--training_pg_master_port",
        type=int,
        default=11111,
        help="The port used for PyTorch's process group initialization across training processes.",
    )
    parser.add_argument(
        "--train_loader_master_port",
        type=int,
        default=11112,
        help="The port used for RPC initialization across all sampling workers of training loader.",
    )
    parser.add_argument(
        "--test_loader_master_port",
        type=int,
        default=11113,
        help="The port used for RPC initialization across all sampling workers of testing loader.",
    )
    args = parser.parse_args()

    f = open("dist_sage_sup.txt", "a+")
    f.write("--- Distributed training example of supervised SAGE ---\n")
    f.write(f"* dataset: {args.dataset}\n")
    f.write(f"* dataset root dir: {args.dataset_root_dir}\n")
    f.write(f"* number of dataset partitions: {args.num_dataset_partitions}\n")
    f.write(f"* total nodes: {args.num_nodes}\n")
    f.write(f"* node rank: {args.node_rank}\n")
    f.write(
        f"* number of training processes per node: {args.num_training_procs}\n"
    )
    f.write(f"* epochs: {args.epochs}\n")
    f.write(f"* batch size: {args.batch_size}\n")
    f.write(f"* master addr: {args.master_addr}\n")
    f.write(
        f"* training process group master port: {args.training_pg_master_port}\n"
    )
    f.write(f"* training loader master port: {args.train_loader_master_port}\n")
    f.write(f"* testing loader master port: {args.test_loader_master_port}\n")

    f.write("--- Loading data partition ...\n")
    root_dir = osp.join(
        osp.dirname(osp.realpath(__file__)), args.dataset_root_dir
    )
    data_pidx = args.node_rank % args.num_dataset_partitions
    node_label_file = osp.join(root_dir, f"{args.dataset}-label", "label.pt")

    train_idx = torch.load(
        osp.join(
            root_dir,
            f"{args.dataset}-train-partitions",
            f"partition{data_pidx}.pt",
        )
    )
    test_idx = torch.load(
        osp.join(
            root_dir,
            f"{args.dataset}-test-partitions",
            f"partition{data_pidx}.pt",
        )
    )
    train_idx.share_memory_()
    test_idx.share_memory_()

    f.write("--- Launching training processes ...\n")

    torch.multiprocessing.spawn(
        run_training_proc,
        args=(
            args.num_nodes,
            args.node_rank,
            args.num_training_procs,
            args.dataset,
            root_dir,
            node_label_file,
            args.in_channel,
            args.out_channel,
            train_idx,
            test_idx,
            args.epochs,
            args.batch_size,
            args.master_addr,
            args.training_pg_master_port,
            args.train_loader_master_port,
            args.test_loader_master_port,
        ),
        nprocs=args.num_training_procs,
        join=True,
    )
