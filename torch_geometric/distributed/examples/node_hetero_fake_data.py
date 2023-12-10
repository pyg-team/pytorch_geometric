from torch_geometric.distributed.dist_context import DistContext
from torch_geometric.distributed.partition import load_partition_info

import argparse
import time

import torch
import torch.distributed
import torch.nn.functional as F

from torch.nn.parallel import DistributedDataParallel

from torch_geometric.distributed import (
    LocalFeatureStore,
    LocalGraphStore,
    DistNeighborLoader,
)
from torch_geometric.nn import HeteroConv, SAGEConv, Linear


class HeteroGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv(
                {
                    ("v0", "e0", "v0"): SAGEConv(-1, hidden_channels),
                    ("v0", "e0", "v1"): SAGEConv((-1, -1), hidden_channels),
                    ("v1", "e0", "v0"): SAGEConv(
                        (-1, -1), hidden_channels, add_self_loops=False
                    ),
                },
                aggr="sum",
            )
            self.convs.append(conv)

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}
        return self.lin(x_dict["v0"])


@torch.no_grad()
def test(model, test_loader):
    model.eval()
    y_pred = []
    y_true = []
    for i, batch in enumerate(test_loader):
        batch_size = batch["v0"].batch_size
        x = model(batch.x_dict, batch.edge_index_dict)
        x = x.argmax(dim=-1, keepdim=True)[:batch_size]
        y_pred.append(x.cpu())
        y_true.append(batch["v0"].y[:batch_size].cpu())
        print(f"---- test():  i={i}, batch={batch} ----")
        del batch
        if i == len(test_loader) - 1:
            print(" ---- dist.barrier ----")
            torch.distributed.barrier()
    y_pred = torch.cat(y_pred, dim=0)
    y_true = torch.cat(y_true, dim=0).unsqueeze(-1)
    test_acc = int((y_pred == y_true).sum()) / y_pred.size(0)

    return float(test_acc)


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
    @torch.no_grad()
    def init_params():
        # Initialize lazy parameters via forwarding a single batch to the model:
        batch = next(iter(train_loader))
        batch = batch.to(torch.device("cpu"), "edge_index")
        model(batch.x_dict, batch.edge_index_dict)

    graph = LocalGraphStore.from_partition(root_dir, node_rank)
    print(f"-------- graph={graph} ")
    edge_attrs = graph.get_all_edge_attrs()
    print(f"------- edge_attrs ={edge_attrs}")
    feature = LocalFeatureStore.from_partition(root_dir, node_rank)
    (
        meta,
        num_partitions,
        partition_idx,
        node_pb,
        edge_pb,
    ) = load_partition_info(root_dir, node_rank)
    print(f"-------- meta={meta}, partition_idx={partition_idx}")

    node_pb_cat = torch.cat(list(node_pb.values()))
    edge_pb_cat = torch.cat(list(edge_pb.values()))

    feature.num_partitions = num_partitions
    feature.partition_idx = partition_idx
    feature.node_feat_pb = node_pb_cat
    feature.edge_feat_pb = edge_pb_cat
    feature.meta = meta

    graph.num_partitions = num_partitions
    graph.partition_idx = partition_idx
    graph.node_pb = node_pb_cat
    graph.edge_pb = edge_pb_cat
    graph.meta = meta
    # generate some fake labels if not saved during partition making
    num_classes = 2
    feature.labels = torch.randint(num_classes, graph.node_pb.size())
    partition_data = (feature, graph)

    # Define model inputs
    # generate input node split for each fake partition
    num_v0_nodes = node_pb["v0"].size(0)
    input_nodes = torch.arange(num_v0_nodes).split(num_v0_nodes // 2)[node_rank]
    # 50/50 train/test split
    input_nodes = input_nodes.split(input_nodes.size(0) // 2)
    train_idx = ("v0", input_nodes[0])
    print("train_idx:", train_idx, train_idx[1].size(0))
    test_idx = ("v0", input_nodes[1])
    print("test_idx:", test_idx, test_idx[1].size(0))

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
    num_workers = 0
    concurrency = 2
    batch_size = 64
    num_layers = 2
    num_neighbors = [10] * num_layers
    async_sampling = False

    # Create distributed neighbor loader for training
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

    # # Create distributed neighbor loader for testing.
    test_loader = DistNeighborLoader(
        data=partition_data,
        num_neighbors=[-1],
        input_nodes=test_idx,
        batch_size=batch_size,
        shuffle=False,
        device=torch.device("cpu"),
        num_workers=num_workers,
        concurrency=concurrency,
        master_addr=master_addr,
        master_port=test_loader_master_port,
        async_sampling=async_sampling,
        current_ctx=current_ctx,
        disjoint=False,
    )

    model = HeteroGNN(
        hidden_channels=64, out_channels=num_classes, num_layers=num_layers
    )

    init_params()

    model = DistributedDataParallel(model, find_unused_parameters=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    print(f"-----------  START TRAINING  ------------- ")
    # Train and test.
    f = open("dist_sage_sup.txt", "a+")
    for epoch in range(0, epochs):
        model.train()
        start = time.time()
        for i, batch in enumerate(train_loader):
            batch_time = time.time()
            optimizer.zero_grad()
            out = model(batch.x_dict, batch.edge_index_dict)
            batch_size = batch["v0"].batch_size
            out = out[:batch_size]
            target = batch["v0"].y[:batch_size]
            loss = F.cross_entropy(out, target)
            loss.backward()
            optimizer.step()
            print(f"x2_worker: batch={batch}, cnt={i}, loss={loss}, time={time.time() - batch_time}")
            if i == len(train_loader) - 1:
                print(" ---- dist.barrier ----")
                torch.distributed.barrier()

        end = time.time()
        f.write(
            f"-- [Trainer {current_ctx.rank}] Epoch: {epoch:03d}, Loss: {loss:.4f}, Epoch Time: {end - start}\n"
        )
        print(
            f"-- [Trainer {current_ctx.rank}] Epoch: {epoch:03d}, Loss: {loss:.4f}, Epoch Time: {end - start}\n"
        )
        print(
            "\n***************************************************************\n"
        )

        # Test accuracy.
        if epoch % 3 == 0:  # or epoch > (epochs // 2):
            test_acc = test(model, test_loader)
            f.write(
                f"-- [Trainer {current_ctx.rank}] Test Accuracy: {test_acc:.4f}\n"
            )
            print(
                f"-- [Trainer {current_ctx.rank}] Test Accuracy: {test_acc:.4f}\n"
            )
        print(
            "\n***************************************************************\n"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Arguments for distributed training of supervised SAGE."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ogbn-mags",
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
        default=12,
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
    root_dir = "/home/pyg/graphlearn-dev/partition_fake"

    f.write("--- Launching training processes ...\n")

    torch.multiprocessing.spawn(
        run_training_proc,
        args=(
            args.num_nodes,
            args.node_rank,
            args.num_training_procs,
            args.dataset,
            root_dir,
            None,
            args.in_channel,
            args.out_channel,
            None,
            None,
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
