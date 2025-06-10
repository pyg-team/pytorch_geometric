"""Single-node, multi-GPU example."""

import argparse
import os
import os.path as osp
import time

import cupy
import rmm
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from cugraph.gnn import (
    cugraph_comms_create_unique_id,
    cugraph_comms_init,
    cugraph_comms_shutdown,
)
from ogb.nodeproppred import PygNodePropPredDataset
from torch.nn.parallel import DistributedDataParallel

import torch_geometric
from torch_geometric import seed_everything
from torch_geometric.utils import (
    add_self_loops,
    remove_self_loops,
    to_undirected,
)

# Allow computation on objects that are larger than GPU memory
# https://docs.rapids.ai/api/cudf/stable/developer_guide/library_design/#spilling-to-host-memory
os.environ['CUDF_SPILL'] = '1'

# Ensures that a CUDA context is not created on import of rapids.
# Allows pytorch to create the context instead
os.environ['RAPIDS_NO_INITIALIZE'] = '1'


def arg_parse():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, )
    parser.add_argument(
        '--dataset',
        type=str,
        default='ogbn-arxiv',
        choices=['ogbn-papers100M', 'ogbn-products', 'ogbn-arxiv'],
        help='Dataset name.',
    )
    parser.add_argument(
        '--dataset_dir',
        type=str,
        default='/workspace/data',
        help='Root directory of dataset.',
    )
    parser.add_argument(
        "--dataset_subdir",
        type=str,
        default="ogbn-arxiv",
        help="directory of dataset.",
    )
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--wd', type=float, default=0.000)
    parser.add_argument('-e', '--epochs', type=int, default=50)
    parser.add_argument('-b', '--batch_size', type=int, default=1024)
    parser.add_argument('--fan_out', type=int, default=10)
    parser.add_argument('--warmup_steps', type=int, default=20)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument(
        '--use_directed_graph',
        action='store_true',
        help='Whether or not to use directed graph',
    )
    parser.add_argument(
        '--add_self_loop',
        action='store_true',
        help='Whether or not to add self loop',
    )
    parser.add_argument(
        "--model",
        type=str,
        default='GCN',
        choices=[
            'SAGE',
            'GAT',
            'GCN',
            # TODO: Uncomment when we add support for disjoint sampling
            # 'SGFormer',
        ],
        help="Model used for training, default GCN",
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=1,
        help="If using GATConv or GT, number of attention heads to use",
    )
    parser.add_argument(
        '--num_devices',
        type=int,
        default=-1,
        help='How many GPUs to use. Defaults to all available GPUs',
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Whether or not to generate statistical report',
    )
    args = parser.parse_args()

    return args


def evaluate(rank, loader, model):
    with torch.no_grad():
        total_correct = total_examples = 0
        for batch in loader:
            batch = batch.to(rank)
            batch_size = batch.batch_size

            batch.y = batch.y.to(torch.long)
            out = model(batch.x, batch.edge_index)[:batch_size]

            pred = out.argmax(dim=-1)
            y = batch.y[:batch_size].view(-1).to(torch.long)

            total_correct += (pred == y).sum()
            total_examples += y.size(0)

        acc = total_correct.item() / total_examples
    return acc


def init_pytorch_worker(rank, world_size, cugraph_id):

    rmm.reinitialize(
        devices=rank,
        managed_memory=True,
        pool_allocator=True,
    )

    cupy.cuda.Device(rank).use()
    from rmm.allocators.cupy import rmm_cupy_allocator

    cupy.cuda.set_allocator(rmm_cupy_allocator)

    import cudf
    cudf.set_option("spill", True)
    torch.cuda.set_device(rank)

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)

    cugraph_comms_init(rank=rank, world_size=world_size, uid=cugraph_id,
                       device=rank)


def run_train(rank, args, data, world_size, cugraph_id, model, split_idx,
              num_classes, wall_clock_start):

    epochs = args.epochs
    batch_size = args.batch_size
    fan_out = args.fan_out
    num_layers = args.num_layers

    init_pytorch_worker(
        rank,
        world_size,
        cugraph_id,
    )

    model = model.to(rank)
    model = DistributedDataParallel(model, device_ids=[rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.wd)

    kwargs = dict(
        num_neighbors=[fan_out] * num_layers,
        batch_size=batch_size,
    )
    from cugraph_pyg.data import GraphStore, TensorDictFeatureStore
    from cugraph_pyg.loader import NeighborLoader

    graph_store = GraphStore(is_multi_gpu=True)
    ixr = torch.tensor_split(data.edge_index, world_size, dim=1)[rank]
    graph_store[dict(
        edge_type=('node', 'rel', 'node'),
        layout='coo',
        is_sorted=False,
        size=(data.num_nodes, data.num_nodes),
    )] = ixr

    feature_store = TensorDictFeatureStore()
    feature_store['node', 'x', None] = data.x
    feature_store['node', 'y', None] = data.y

    dist.barrier()

    ix_train = torch.tensor_split(split_idx['train'], world_size)[rank].cuda()
    train_loader = NeighborLoader(
        (feature_store, graph_store),
        input_nodes=ix_train,
        shuffle=True,
        drop_last=True,
        **kwargs,
    )

    ix_val = torch.tensor_split(split_idx['valid'], world_size)[rank].cuda()
    val_loader = NeighborLoader(
        (feature_store, graph_store),
        input_nodes=ix_val,
        drop_last=True,
        **kwargs,
    )

    ix_test = torch.tensor_split(split_idx['test'], world_size)[rank].cuda()
    test_loader = NeighborLoader(
        (feature_store, graph_store),
        input_nodes=ix_test,
        drop_last=True,
        local_seeds_per_call=80000,
        **kwargs,
    )

    dist.barrier()

    warmup_steps = args.warmup_steps
    dist.barrier()
    torch.cuda.synchronize()

    if rank == 0:
        prep_time = time.perf_counter() - wall_clock_start
        print("Total time before training begins (prep_time) =", prep_time,
              "seconds")
        print("Beginning training...")

    val_accs = []
    times = []
    train_times = []
    inference_times = []
    best_val = 0.
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        train_start = time.perf_counter()
        total_loss = 0
        i = 0
        for i, batch in enumerate(train_loader):
            if i == warmup_steps:
                torch.cuda.synchronize()
            batch = batch.to(rank)
            batch_size = batch.batch_size
            batch.y = batch.y.to(torch.long)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)
            loss = F.cross_entropy(out[:batch_size], batch.y[:batch_size])
            loss.backward()
            optimizer.step()
            total_loss += loss
        train_end = time.perf_counter()
        train_times.append(train_end - train_start)
        nb = i + 1.0
        total_loss /= nb
        dist.barrier()
        torch.cuda.synchronize()

        inference_start = time.perf_counter()
        train_acc = evaluate(rank, train_loader, model)
        dist.barrier()
        val_acc = evaluate(rank, val_loader, model)
        dist.barrier()

        inference_times.append(time.perf_counter() - inference_start)
        val_accs.append(val_acc)
        if rank == 0:
            print(f'Epoch {epoch:02d}, Loss: {total_loss:.4f}, Approx. Train:'
                  f' {train_acc:.4f} Time: {train_end - train_start:.4f}s')
            print(f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, ')

        times.append(time.perf_counter() - train_start)
        if val_acc > best_val:
            best_val = val_acc

    print(f'Total time used for rank: {rank:02d} is '
          f'{time.perf_counter()-start:.4f}')
    if rank == 0:
        val_acc = torch.tensor(val_accs)
        print('============================')
        print("Average Epoch Time on training: {:.4f}".format(
            torch.tensor(train_times).mean()))
        print("Average Epoch Time on inference: {:.4f}".format(
            torch.tensor(inference_times).mean()))
        print(f"Average Epoch Time: {torch.tensor(times).mean():.4f}")
        print(f"Median time per epoch: {torch.tensor(times).median():.4f}s")
        print(f'Final Validation: {val_acc.mean():.4f} ± {val_acc.std():.4f}')
        print(f"Best validation accuracy: {best_val:.4f}")

    if rank == 0:
        print("Testing...")
    final_test_acc = evaluate(rank, test_loader, model)
    dist.barrier()
    if rank == 0:
        print(f'Test Accuracy: {final_test_acc:.4f} for rank: {rank:02d}')
    if rank == 0:
        total_time = time.perf_counter() - wall_clock_start
        print(f"Total Training Runtime: {total_time - prep_time}s")
        print(f"Total Program Runtime: {total_time}s")

    cugraph_comms_shutdown()
    dist.destroy_process_group()


if __name__ == '__main__':

    args = arg_parse()
    seed_everything(123)
    wall_clock_start = time.perf_counter()

    root = osp.join(args.dataset_dir, args.dataset_subdir)
    dataset = PygNodePropPredDataset(name=args.dataset, root=root)
    split_idx = dataset.get_idx_split()
    data = dataset[0]
    if not args.use_directed_graph:
        data.edge_index = to_undirected(data.edge_index, reduce="mean")
    if args.add_self_loop:
        data.edge_index, _ = remove_self_loops(data.edge_index)
        data.edge_index, _ = add_self_loops(data.edge_index,
                                            num_nodes=data.num_nodes)
    data.y = data.y.reshape(-1)

    print(f"Training {args.dataset} with {args.model} model.")
    if args.model == "GAT":
        model = torch_geometric.nn.models.GAT(dataset.num_features,
                                              args.hidden_channels,
                                              args.num_layers,
                                              dataset.num_classes,
                                              heads=args.num_heads)
    elif args.model == "GCN":
        model = torch_geometric.nn.models.GCN(
            dataset.num_features,
            args.hidden_channels,
            args.num_layers,
            dataset.num_classes,
        )
    elif args.model == "SAGE":
        model = torch_geometric.nn.models.GraphSAGE(
            dataset.num_features,
            args.hidden_channels,
            args.num_layers,
            dataset.num_classes,
        )
    elif args.model == 'SGFormer':
        # TODO add support for this with disjoint sampling
        model = torch_geometric.nn.models.SGFormer(
            in_channels=dataset.num_features,
            hidden_channels=args.hidden_channels,
            out_channels=dataset.num_classes,
            trans_num_heads=args.num_heads,
            trans_dropout=args.dropout,
            gnn_num_layers=args.num_layers,
            gnn_dropout=args.dropout,
        )
    else:
        raise ValueError(f'Unsupported model type: {args.model}')

    print("Data =", data)
    if args.num_devices < 1:
        world_size = torch.cuda.device_count()
    elif args.num_devices <= torch.cuda.device_count():
        world_size = args.num_devices
    else:
        world_size = torch.cuda.device_count()
    print('Let\'s use', world_size, 'GPUs!')

    # Create the uid needed for cuGraph comms
    cugraph_id = cugraph_comms_create_unique_id()

    if world_size > 1:
        mp.spawn(
            run_train,
            args=(args, data, world_size, cugraph_id, model, split_idx,
                  dataset.num_classes, wall_clock_start),
            nprocs=world_size,
            join=True,
        )
    else:
        run_train(0, args, data, world_size, cugraph_id, model, split_idx,
                  dataset.num_classes, wall_clock_start)

    total_time = round(time.perf_counter() - wall_clock_start, 2)
    print("Total Program Runtime (total_time) =", total_time, "seconds")
