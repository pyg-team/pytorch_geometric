import torch_geometric.distributed as pyg_dist
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
from torch_geometric.nn import GraphSAGE

from torch_geometric.distributed import LocalFeatureStore, LocalGraphStore
from torch_geometric.sampler import NegativeSampling

print("\n\n\n\n\n\n")


@torch.no_grad()
def test(model, test_loader, dataset_name):
    evaluator = Evaluator(name=dataset_name)
    model.eval()
    xs = []
    y_true = []
    for i, batch in enumerate(test_loader):
        if i == 0:
            device = batch.x.device
        x = model(batch.x, batch.edge_index)[: batch.batch_size]
        xs.append(x.cpu())
        y_true.append(batch.y[: batch.batch_size].cpu())
        print(f"---- test():  i={i}, batch={batch} ----")
        del batch
        if i == len(test_loader) - 1:
            print(" ---- dist.barrier ----")
            torch.distributed.barrier()
    xs = [t.to(device) for t in xs]
    y_true = [t.to(device) for t in y_true]
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
    print(f"-------- graph={graph} ")
    edge_attrs = graph.get_all_edge_attrs()
    print(f"-----777---- edge_attrs ={edge_attrs}")

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
    print(f" ----- partition_data={partition_data[0]}  ")

    # Initialize graphlearn_torch distributed worker group context.
    current_ctx = DistContext(
        world_size=num_nodes * num_training_procs_per_node,
        rank=node_rank * num_training_procs_per_node + local_proc_rank,
        global_world_size=num_nodes * num_training_procs_per_node,
        global_rank=node_rank * num_training_procs_per_node + local_proc_rank,
        group_name="distributed-sage-supervised-trainer",
    )
    current_device = torch.device("cpu")

    print(
        f"----------- 111.2 -------------local_proc_rank={local_proc_rank}, current_ctx.rank={current_ctx.rank}, current_ctx.world_size={current_ctx.world_size} "
    )
    # Initialize training process group of PyTorch.
    torch.distributed.init_process_group(
        backend="gloo",
        rank=current_ctx.rank,
        world_size=current_ctx.world_size,
        init_method="tcp://{}:{}".format(master_addr, training_pg_master_port),
    )

    print(f"----------- 222 ------------- ")
    # Create distributed neighbor loader for training
    train_idx = train_idx.split(
        train_idx.size(0) // num_training_procs_per_node
    )[local_proc_rank]

    edge_label_index_train = (
        None,
        torch.stack(
            [
                graph.get_edge_index((None, "coo"))[0],
                graph.get_edge_index((None, "coo"))[1],
            ],
            dim=0,
        ),
    )
    num_workers = 2
    concurrency = 4
    train_loader = pyg_dist.DistLinkNeighborLoader(
        data=partition_data,
        num_workers=num_workers,
        num_neighbors=[15, 10, 5],
        batch_size=batch_size,
        edge_label_index=edge_label_index_train,  # dataset.data.edge_index, ##TODO ???
        edge_label=None,
        edge_label_time=None,
        replace=False,
        directed=True,
        disjoint=False,
        temporal_strategy="uniform",
        neg_sampling=NegativeSampling("triplet"),  # TODO support 'triplet'
        neg_sampling_ratio=1,
        time_attr=None,
        transform=None,
        is_sorted=False,
        filter_per_worker=False,
        shuffle=True,
        drop_last=False,
        device=current_device,
        async_sampling=True,
        master_addr=master_addr,
        master_port=train_loader_master_port,
        concurrency=concurrency,
        current_ctx=current_ctx,
    )

    print(f"----------- 333 ------------- ")
    # below is same as train edge_label_index - TODO split into train test
    edge_label_index_test = (
        None,
        torch.stack(
            [
                graph.get_edge_index((None, "coo"))[0],
                graph.get_edge_index((None, "coo"))[1],
            ],
            dim=0,
        ),
    )

    # Create distributed neighbor loader for testing.
    test_idx = test_idx.split(test_idx.size(0) // num_training_procs_per_node)[
        local_proc_rank
    ]
    test_loader = pyg_dist.DistLinkNeighborLoader(
        data=partition_data,
        num_neighbors=[15, 10, 5],
        num_workers=num_workers,
        batch_size=batch_size,
        edge_label_index=edge_label_index_test,  # dataset.data.edge_index, ##TODO ???
        edge_label=None,
        edge_label_time=None,
        replace=False,
        directed=True,
        disjoint=True,
        temporal_strategy="uniform",
        neg_sampling=NegativeSampling("triplet"),  # TODO support 'triplet'
        neg_sampling_ratio=1,
        time_attr=None,
        transform=None,
        is_sorted=False,
        filter_per_worker=False,
        shuffle=False,
        drop_last=False,
        device=current_device,
        async_sampling=True,
        master_addr=master_addr,
        master_port=test_loader_master_port,
        concurrency=concurrency,
        current_ctx=current_ctx,
    )

    # Define model and optimizer.
    # torch.cuda.set_device(current_device)
    model = GraphSAGE(
        in_channels=in_channels,
        hidden_channels=256,
        num_layers=3,
        out_channels=out_channels,
    ).to(current_device)
    model = DistributedDataParallel(
        model
    )  # , device_ids=[current_device.index])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    print(f"----------- 444 ------------- ")
    # Train and test.
    f = open("dist_sage_sup.txt", "a+")
    for epoch in range(0, epochs):
        model.train()
        start = time.time()
        cnt = 0
        for i, batch in enumerate(train_loader):
            print(f"-------- x2_worker: batch={batch}, cnt={i} --------- ")
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)[
                : batch.batch_size
            ].log_softmax(dim=-1)
            loss = F.nll_loss(out, batch.y[: batch.batch_size])
            loss.backward()
            optimizer.step()
            if i == len(test_loader) - 1:
                print(" ---- dist.barrier ----")
                torch.distributed.barrier()

        end = time.time()
        f.write(
            f"-- [Trainer {current_ctx.rank}] Epoch: {epoch:03d}, Loss: {loss:.4f}, Epoch Time: {end - start}\n"
        )
        print(
            f"-- [Trainer {current_ctx.rank}] Epoch: {epoch:03d}, Loss: {loss:.4f}, Epoch Time: {end - start}\n"
        )
        print("\n\n\n\n\n\n")
        print(
            "********************************************************************************************** "
        )
        print("\n\n\n\n\n\n")

        # Test accuracy.
        # if epoch == 0 or epoch > (epochs // 2):
        if epoch % 1 == 0:  # or epoch > (epochs // 2):
            test_acc = test(model, test_loader, dataset_name)
            f.write(
                f"-- [Trainer {current_ctx.rank}] Test Accuracy: {test_acc:.4f}\n"
            )
            print(
                f"-- [Trainer {current_ctx.rank}] Test Accuracy: {test_acc:.4f}\n"
            )

            print("\n\n\n\n\n\n")
            print(
                "********************************************************************************************** "
            )
            print("\n\n\n\n\n\n")
            torch.cuda.synchronize()
            torch.distributed.barrier()

    print(f"----------- 555 ------------- ")


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
        default=3,
        help="The number of training epochs.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
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
        default=33333,
        help="The port used for PyTorch's process group initialization across training processes.",
    )
    parser.add_argument(
        "--train_loader_master_port",
        type=int,
        default=33334,
        help="The port used for RPC initialization across all sampling workers of training loader.",
    )
    parser.add_argument(
        "--test_loader_master_port",
        type=int,
        default=33335,
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
    r"""
  dataset = pyg_dist.DistDataset()
  dataset.load(
    root_dir=osp.join(root_dir, f'{args.dataset}-partitions'),
    partition_idx=data_pidx,
    node_label_file=osp.join(root_dir, f'{args.dataset}-label', 'label.pt'),
    partition_format="pyg"
  )
  #graph_store1 = LocalGraphStore()
  graph_store1 = LocalGraphStore.from_partition(osp.join(root_dir, f'{args.dataset}-partitions'), data_pidx)
  print(f"-------- graph_store1={graph_store1} ")
  edge_attrs = graph_store1.get_all_edge_attrs()
  print(f"-----777---- edge_attrs ={edge_attrs}")


  #feature_store1 = LocalFeatureStore.from_partition(osp.join(root_dir, f'{args.dataset}-partitions'), data_pidx)
  #print(f"-------- feature_store1={feature_store1} ")


  #dataset1 = (graph_store1, feature_store1)
  """
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
    # print(f"---- dataset={dataset}, train_idx={train_idx}, test_idx={test_idx} ")

    # torch.multiprocessing.set_start_method('spawn')
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
