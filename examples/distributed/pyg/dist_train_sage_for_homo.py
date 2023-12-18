import argparse
import os
import os.path as osp
import time

import torch
import torch.distributed
import torch.nn.functional as F
from ogb.nodeproppred import Evaluator
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

import torch_geometric.distributed as pyg_dist
from torch_geometric.distributed import LocalFeatureStore, LocalGraphStore
from torch_geometric.distributed.dist_context import DistContext, DistRole
from torch_geometric.distributed.partition import load_partition_info
from torch_geometric.nn import GraphSAGE
from torch_geometric.typing import Tuple


@torch.no_grad()
def test(model, loader):
    model.eval()
    total_examples = total_correct = 0
    for i, batch in enumerate(loader):
        batch_time_start = time.time()
        batch_size = batch.batch_size
        out = model(batch.x, batch.edge_index)[:batch_size]
        pred = out.argmax(dim=-1)
        pred = int((pred == batch.y[:batch_size]).sum())
        batch_acc = pred / batch_size
        total_examples += batch_size
        total_correct += pred
    torch.distributed.barrier()

    return total_correct / total_examples


def run_training_proc(local_proc_rank: int, num_nodes: int, node_rank: int,
                      num_training_procs_per_node: int, dataset_name: str,
                      root_dir: str, node_label_file: str, in_channels: int,
                      out_channels: int, train_idx: torch.Tensor,
                      test_idx: torch.Tensor, epochs: int, num_neighbors: str,
                      batch_size: int, num_workers: int, concurrency: int,
                      learning_rate: float, master_addr: str,
                      training_pg_master_port: int,
                      train_loader_master_port: int,
                      test_loader_master_port: int):

    # load partition into graph
    graph = LocalGraphStore.from_partition(
        osp.join(root_dir, f'{dataset_name}-partitions'), node_rank)
    edge_attrs = graph.get_all_edge_attrs()

    print(f"-----  000 edge_attrs={edge_attrs} ")

    # load partition into feature
    feature = LocalFeatureStore.from_partition(
        osp.join(root_dir, f'{dataset_name}-partitions'), node_rank)

    # load partition information
    (meta, num_partitions, partition_idx, node_pb,
     edge_pb) = load_partition_info(
         osp.join(root_dir, f'{dataset_name}-partitions'), node_rank)

    # setup the partition information in graph
    graph.num_partitions = num_partitions
    graph.partition_idx = partition_idx
    graph.node_pb = node_pb
    graph.edge_pb = edge_pb
    graph.meta = meta

    # setup the partition information in feature
    feature.num_partitions = num_partitions
    feature.partition_idx = partition_idx
    feature.node_feat_pb = node_pb
    feature.edge_feat_pb = edge_pb
    feature.feature_pb = node_pb
    feature.meta = meta

    print(
        f"-------- meta={meta}, partition_idx={partition_idx}, node_pb={node_pb} "
    )

    # load the label file and put into graph as labels
    if node_label_file is not None:
        if isinstance(node_label_file, dict):
            whole_node_labels = {}
            for ntype, file in node_label_file.items():
                whole_node_labels[ntype] = torch.load(file)
        else:
            whole_node_labels = torch.load(node_label_file)
    feature.labels = whole_node_labels

    partition_data = (feature, graph)

    # Initialize distributed context.
    current_ctx = DistContext(
        world_size=num_nodes * num_training_procs_per_node,
        rank=node_rank * num_training_procs_per_node + local_proc_rank,
        global_world_size=num_nodes * num_training_procs_per_node,
        global_rank=node_rank * num_training_procs_per_node + local_proc_rank,
        group_name='distributed-sage-supervised-trainer')
    current_device = torch.device('cpu')
    rpc_worker_names = {}

    # Initialize DDP training process group.
    torch.distributed.init_process_group(
        backend='gloo', rank=current_ctx.rank,
        world_size=current_ctx.world_size,
        init_method='tcp://{}:{}'.format(master_addr, training_pg_master_port))

    # setup the train seeds for the loader
    train_idx = train_idx.split(
        train_idx.size(0) // num_training_procs_per_node)[local_proc_rank]

    num_neighbors = num_neighbors.split(",")
    num_neighbors = [int(i) for i in num_neighbors]
    # Create distributed neighbor loader for training
    train_loader = pyg_dist.DistNeighborLoader(
        data=partition_data,
        num_neighbors=num_neighbors,
        input_nodes=train_idx,
        batch_size=batch_size,
        shuffle=True,
        device=torch.device('cpu'),
        num_workers=num_workers,
        concurrency=concurrency,
        master_addr=master_addr,
        master_port=train_loader_master_port,
        async_sampling=True,
        filter_per_worker=False,
        current_ctx=current_ctx,
        drop_last=True,
    )

    # setup the train seeds for the loader
    test_idx = test_idx.split(test_idx.size(0) //
                              num_training_procs_per_node)[local_proc_rank]

    # Create distributed neighbor loader for testing.
    test_loader = pyg_dist.DistNeighborLoader(
        data=partition_data,
        num_neighbors=num_neighbors,
        input_nodes=test_idx,
        batch_size=batch_size,
        shuffle=False,
        device=torch.device('cpu'),
        num_workers=num_workers,
        concurrency=concurrency,
        master_addr=master_addr,
        master_port=test_loader_master_port,
        async_sampling=True,
        filter_per_worker=False,
        current_ctx=current_ctx,
        drop_last=True,
    )

    # Define model and optimizer.
    model = GraphSAGE(
        in_channels=in_channels,
        hidden_channels=256,
        num_layers=3,
        out_channels=out_channels,
    ).to(current_device)
    model = DistributedDataParallel(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    res = f'./res/partitions{num_nodes}_batchsize{batch_size}_lr{learning_rate}'
    os.makedirs(res, exist_ok=True)

    # Train and test.
    f = open(f'{res}/dist_train_sage_for_homo_rank{node_rank}.txt', 'a+')
    for epoch in range(0, epochs):
        model.train()
        pbar = tqdm(total=train_idx.size(0))
        start = time.time()
        times = []
        for i, batch in enumerate(train_loader):
            if i == 0:
                pbar.set_description(f'Epoch {epoch:02d}')
            batch_time_start = time.time()
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)[:batch.batch_size]
            loss = F.cross_entropy(out, batch.y[:batch.batch_size])
            loss.backward()
            optimizer.step()
            batch_time = time.time() - batch_time_start
            times.append(batch_time)
            print(
                f"-------- dist_training: i={i}, batch loss={loss}, batch_time={batch_time} --------- "
            )
            pbar.update(batch_size)
        pbar.close()
        torch.distributed.barrier()

        end = time.time()
        f.write(
            f'-- [Trainer {current_ctx.rank}] Epoch: {epoch:03d}, Loss: {loss:.4f}, Epoch Time: {end - start}\n'
        )
        print(
            f'-- [Trainer {current_ctx.rank}] Epoch: {epoch:03d}, Loss: {loss:.4f}, Epoch Time: {end - start}\n'
        )
        print("\n")
        print(
            "********************************************************************************************** "
        )
        print("\n")

        # Test accuracy.
        if epoch % 5 == 0 and epoch > 0:
            print("  Test ... ")
            test_acc = test(model, test_loader)
            torch.distributed.barrier()
            f.write(
                f'-- [Trainer {current_ctx.rank}] Test Accuracy: {test_acc:.4f}\n'
            )
            print(
                f'-- [Trainer {current_ctx.rank}] Test Accuracy: {test_acc:.4f}\n'
            )

            print("\n\n")
            print(
                "********************************************************************************************** "
            )
        print("\n\n")
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Arguments for distributed training of supervised SAGE.")
    parser.add_argument(
        "--dataset",
        type=str,
        default='ogbn-products',
        help="The name of ogbn dataset.",
    )
    parser.add_argument(
        "--in_channel", type=int, default=100,
        help="in channel of the dataset, default is for ogbn-products")
    parser.add_argument(
        "--out_channel", type=int, default=47,
        help="out channel of the dataset, default is for ogbn-products")
    parser.add_argument(
        "--dataset_root_dir",
        type=str,
        default='../../data/products',
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
        default=6,
        help="The number of training epochs.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1024,
        help="Batch size for the training and testing dataloader.",
    )
    parser.add_argument(
        "--num_neighbors",
        type=str,
        default="15,10,5",
        help="number of the layers",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="number of sampler workers.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=2,
        help="concurrency number for mp.queue to send the sampler output.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.0004,
        help="learning rate",
    )

    parser.add_argument(
        "--master_addr",
        type=str,
        default='localhost',
        help="The master address for RPC initialization.",
    )
    parser.add_argument(
        "--training_pg_master_port",
        type=int,
        default=11111,
        help=
        "The port used for PyTorch's process group initialization across training processes.",
    )
    parser.add_argument(
        "--train_loader_master_port",
        type=int,
        default=11112,
        help=
        "The port used for RPC initialization across all sampling workers of training loader.",
    )
    parser.add_argument(
        "--test_loader_master_port",
        type=int,
        default=11113,
        help=
        "The port used for RPC initialization across all sampling workers of testing loader.",
    )
    args = parser.parse_args()

    f = open('dist_train_sage_for_homo.txt', 'a+')
    f.write('--- Distributed training example of supervised SAGE ---\n')
    f.write(f'* dataset: {args.dataset}\n')
    f.write(f'* dataset root dir: {args.dataset_root_dir}\n')
    f.write(f'* number of dataset partitions: {args.num_dataset_partitions}\n')
    f.write(f'* total nodes: {args.num_nodes}\n')
    f.write(f'* node rank: {args.node_rank}\n')
    f.write(
        f'* number of training processes per node: {args.num_training_procs}\n'
    )
    f.write(f'* epochs: {args.epochs}\n')
    f.write(f'* batch size: {args.batch_size}\n')
    f.write(f'* number of sampler workers: {args.num_workers}\n')
    f.write(f'* concurrency number for mp.queue: {args.concurrency}\n')
    f.write(f'* master addr: {args.master_addr}\n')
    f.write(
        f'* training process group master port: {args.training_pg_master_port}\n'
    )
    f.write(
        f'* training loader master port: {args.train_loader_master_port}\n')
    f.write(f'* testing loader master port: {args.test_loader_master_port}\n')

    f.write('--- Loading data partition ...\n')
    root_dir = osp.join(osp.dirname(osp.realpath(__file__)),
                        args.dataset_root_dir)
    data_pidx = args.node_rank % args.num_dataset_partitions

    node_label_file = osp.join(root_dir, f'{args.dataset}-label', 'label.pt')

    train_idx = torch.load(
        osp.join(root_dir, f'{args.dataset}-train-partitions',
                 f'partition{data_pidx}.pt'))
    test_idx = torch.load(
        osp.join(root_dir, f'{args.dataset}-test-partitions',
                 f'partition{data_pidx}.pt'))

    f.write('--- Launching training processes ...\n')

    torch.multiprocessing.spawn(
        run_training_proc,
        args=(args.num_nodes, args.node_rank, args.num_training_procs,
              args.dataset, root_dir, node_label_file, args.in_channel,
              args.out_channel, train_idx, test_idx, args.epochs,
              args.num_neighbors, args.batch_size, args.num_workers,
              args.concurrency, args.learning_rate, args.master_addr,
              args.training_pg_master_port, args.train_loader_master_port,
              args.test_loader_master_port), nprocs=args.num_training_procs,
        join=True)
