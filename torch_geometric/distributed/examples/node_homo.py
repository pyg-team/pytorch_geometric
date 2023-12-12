import torch_geometric.distributed as pyg_dist
from torch_geometric.typing import Tuple
from torch_geometric.distributed.dist_context import DistContext, DistRole
from torch_geometric.distributed.partition import load_partition_info


import argparse
import os.path as osp
import os
import time

import torch
import torch.distributed
import torch.nn.functional as F

from ogb.nodeproppred import Evaluator
from torch.nn.parallel import DistributedDataParallel
from torch_geometric.nn import GraphSAGE

from torch_geometric.distributed import LocalFeatureStore, LocalGraphStore
import matplotlib.pyplot as plt

import logging

logging.basicConfig(
    format='%(levelname)s:%(process)d:%(message)s', level=logging.INFO
)


@torch.no_grad()
def test(model, loader, epoch, res):
    model.eval()
    total_examples = total_correct = 0
    with open(f"{res}/test.csv", "a+") as test_file:
        for i, batch in enumerate(loader):
            batch_time_start = time.time()
            batch_size = batch.batch_size
            out = model(batch.x, batch.edge_index)[:batch_size]
            pred = out.argmax(dim=-1)
            pred = int((pred == batch.y[:batch_size]).sum())
            batch_acc = pred / batch_size
            total_examples += batch_size
            total_correct += pred
            batch_time = time.time() - batch_time_start
            print(
                f"---- test():  i={i}, batch_acc: {batch_acc}, batch_time={batch_time} ----"
            )
            test_file.write(f"{epoch},{i},{batch_acc},{batch_time}\n")
        torch.distributed.barrier()

    return total_correct / total_examples


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
    print(f"-------- graph={graph} ")
    edge_attrs = graph.get_all_edge_attrs()
    print(f"------- edge_attrs ={edge_attrs}")
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
    print(
        f"-------- meta={meta}, partition_idx={partition_idx}, node_pb={node_pb} "
    )

    graph.num_partitions = num_partitions
    graph.partition_idx = partition_idx
    graph.node_pb = node_pb
    graph.edge_pb = edge_pb
    graph.meta = meta

    feature.num_partitions = num_partitions
    feature.partition_idx = partition_idx
    feature.node_feat_pb = node_pb
    feature.edge_feat_pb = edge_pb
    feature.meta = meta
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
    current_device = torch.device("cpu")

    # Create distributed neighbor loader for training
    train_idx = train_idx.split(
        train_idx.size(0) // num_training_procs_per_node
    )[local_proc_rank]

    # Create distributed neighbor loader for testing.
    test_idx = test_idx.split(test_idx.size(0) // num_training_procs_per_node)[
        local_proc_rank
    ]

    # Initialize training process group of PyTorch.
    torch.distributed.init_process_group(
        backend="gloo",
        rank=current_ctx.rank,
        world_size=current_ctx.world_size,
        init_method="tcp://{}:{}".format(master_addr, training_pg_master_port),
    )
    num_workers = [4, 0]
    concurrency = 10,
    async_sampling = [True, False]
    for nw in num_workers:
        for c in concurrency:
            for a in async_sampling:
                res = f'./res/nw{nw}c{c}async{a}'
                os.makedirs(res, exist_ok=True)
                # Create loaders
                train_loader = pyg_dist.DistNeighborLoader(
                    data=partition_data,
                    num_neighbors=[15, 10, 5],
                    input_nodes=train_idx,
                    batch_size=8192,
                    shuffle=True,
                    device=torch.device("cpu"),
                    num_workers=nw,
                    concurrency=c,
                    master_addr=master_addr,
                    master_port=11112,
                    async_sampling=a,
                    current_ctx=current_ctx,
                    drop_last=True,
                )

                test_loader = pyg_dist.DistNeighborLoader(
                    data=partition_data,
                    num_neighbors=[15, 10, 5],
                    input_nodes=test_idx,
                    batch_size=8192,
                    shuffle=False,
                    device=torch.device("cpu"),
                    num_workers=nw,
                    concurrency=c,
                    master_addr=master_addr,
                    master_port=11113,
                    async_sampling=a,
                    current_ctx=current_ctx,
                    drop_last=True,
                )

                model = GraphSAGE(
                    in_channels=100,
                    hidden_channels=256,
                    num_layers=3,
                    out_channels=47,
                ).to(current_device)
                model = DistributedDataParallel(model)
                optimizer = torch.optim.Adam(model.parameters(), lr=0.0004)
                # Train and test.
                train_loss = []
                test_accs = []
                test_it = []
                times = []
                for epoch in range(0, epochs):
                    model.train()
                    start = time.time()
                    with open(f"{res}/train.csv", "a+") as train_file:
                        for i, batch in enumerate(train_loader):
                            batch_time_start = time.time()
                            optimizer.zero_grad()
                            out = model(batch.x, batch.edge_index)[
                                : batch.batch_size
                            ]
                            loss = F.cross_entropy(
                                out, batch.y[: batch.batch_size]
                            )
                            train_loss.append(loss)
                            loss.backward()
                            optimizer.step()
                            batch_time = time.time() - batch_time_start
                            times.append(batch_time)
                            print(
                                f"-------- dist_train_2nodes: i={i}, batch loss={loss}, batch_time={batch_time} --------- "
                            )
                            train_file.write(
                                f"{epoch},{i},{loss},{batch_time}\n"
                            )
                        epoch_time = time.time() - start
                        torch.distributed.barrier()
                    print(
                        f"-- [Trainer {current_ctx.rank}] Epoch: {epoch:03d}, Epoch Loss: {loss:.4f}, Epoch Time: {epoch_time}\n"
                    )
                    print("\n**************\n")
                    # Test accuracy
                    if epoch % 20 == 0:
                        with open(f"{res}/acc.csv", "a+") as acc_file:
                            test_acc = test(model, test_loader, epoch, res)
                            test_it.append(i)
                            test_accs.append(test_acc)
                            acc_file.write(f"{epoch},{test_acc}\n")
                            print(
                                f"-- [Trainer {current_ctx.rank}] Epoch: {epoch:03d} Test Accuracy: {test_acc:.4f}\n"
                            )
                            print("\n**************\n")

    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Arguments for distributed training of supervised SAGE."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ogbn-products",
        help="The name of ogbn dataset.",
    )
    parser.add_argument(
        "--in_channel",
        type=int,
        default=100,
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
        default="../../data/products",
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
        default=10,
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

    node_rank = int(os.getenv('RANK'))
    num_nodes = 4

    torch.multiprocessing.spawn(
        run_training_proc,
        args=(
            num_nodes,
            node_rank,
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
