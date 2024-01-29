import argparse
import os
import tempfile
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from ogb.nodeproppred import PygNodePropPredDataset
from torch.distributed.algorithms.join import Join
from torch.nn.parallel import DistributedDataParallel
from torchmetrics import Accuracy

import torch_geometric

# Allow computation on objects that are larger than GPU memory
# https://docs.rapids.ai/api/cudf/stable/developer_guide/library_design/#spilling-to-host-memory
os.environ['CUDF_SPILL'] = '1'

# Ensures that a CUDA context is not created on import of rapids.
# Allows pytorch to create the context instead
os.environ['RAPIDS_NO_INITIALIZE'] = '1'


def start_dask_cluster():
    from cugraph.testing.mg_utils import enable_spilling
    from dask_cuda import LocalCUDACluster

    cluster = LocalCUDACluster(
        protocol="tcp",
        rmm_pool_size=None,
        memory_limit=None,
        rmm_async=True,
    )

    from dask.distributed import Client
    client = Client(cluster)
    client.wait_for_workers(n_workers=len(cluster.workers))
    client.run(enable_spilling)

    print("Dask Cluster Setup Complete")
    return client, cluster


def shutdown_dask_client(client):
    from cugraph.dask.comms import comms as Comms
    Comms.destroy()
    client.close()


def pyg_num_work(world_size):
    num_work = None
    if hasattr(os, "sched_getaffinity"):
        try:
            num_work = len(os.sched_getaffinity(0)) / (2 * world_size)
        except Exception:
            pass
    if num_work is None:
        num_work = os.cpu_count() / (2 * world_size)
    return int(num_work)


def init_pytorch_worker(rank, world_size, cugraph_data_loader=False):
    if cugraph_data_loader:
        import rmm
        if rank > 0:
            rmm.reinitialize(devices=rank)

        import cupy
        cupy.cuda.Device(rank).use()
        from rmm.allocators.cupy import rmm_cupy_allocator
        cupy.cuda.set_allocator(rmm_cupy_allocator)

        from cugraph.testing.mg_utils import enable_spilling
        enable_spilling()

        torch.cuda.set_device(rank)

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)


def run_train(rank, data, world_size, model, epochs, batch_size, fan_out,
              split_idx, num_classes, cugraph_data_loader, wall_clock_start,
              tempdir=None, num_layers=3):

    init_pytorch_worker(
        rank,
        world_size,
        cugraph_data_loader=cugraph_data_loader,
    )

    if cugraph_data_loader:
        if rank == 0:
            client, cluster = start_dask_cluster()
            from cugraph.dask.comms import comms as Comms
            Comms.initialize(p2p=True)
    else:
        split_idx['train'] = split_idx['train'].split(
            split_idx['train'].size(0) // world_size, dim=0)[rank].clone()
        split_idx['valid'] = split_idx['valid'].split(
            split_idx['valid'].size(0) // world_size, dim=0)[rank].clone()
        split_idx['test'] = split_idx['test'].split(
            split_idx['test'].size(0) // world_size, dim=0)[rank].clone()
    model = model.to(rank)
    model = DistributedDataParallel(model, device_ids=[rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01,
                                 weight_decay=0.0005)

    kwargs = dict(
        num_neighbors=[fan_out] * num_layers,
        batch_size=batch_size,
    )
    # Set Up Neighbor Loading
    if cugraph_data_loader:
        import cugraph
        from cugraph_pyg.data import CuGraphStore
        from cugraph_pyg.loader import BulkSampleLoader
        G = {("N", "E", "N"): data.edge_index}
        N = {"N": data.num_nodes}
        fs = cugraph.gnn.FeatureStore(backend="torch")
        fs.add_data(data.x, "N", "x")
        fs.add_data(data.y, "N", "y")
        dist.barrier()

        if rank == 0:
            print("Rank 0 creating its cugraph store and \
                initializing distributed graph")
            cugraph_store = CuGraphStore(fs, G, N, multi_gpu=True)
            print("Distributed graph initialization complete.")

        if rank != 0:
            print(f"Rank {rank} waiting for distributed graph initialization")
        dist.barrier()

        if rank != 0:
            print(f"Rank {rank} proceeding with store creation")
            cugraph_store = CuGraphStore(fs, {
                k: len(v)
                for k, v in G.items()
            }, N, multi_gpu=False)
            print(f"Rank {rank} created store")
        dist.barrier()

        if rank == 0:
            # Direct cuGraph to sample offline prior to the training loop
            # Sampling will occur in parallel but will be initiated on rank 0
            for epoch in range(epochs):
                train_path = os.path.join(tempdir, f'samples_{epoch}')
                os.mkdir(train_path)
                BulkSampleLoader(cugraph_store, cugraph_store,
                                 input_nodes=split_idx['train'],
                                 directory=train_path, shuffle=True,
                                 drop_last=True, **kwargs)

                print('validation', len(split_idx['valid']))
                eval_path = os.path.join(tempdir, f'samples_eval_{epoch}')
                BulkSampleLoader(cugraph_store, cugraph_store,
                                 input_nodes=split_idx['valid'],
                                 directory=eval_path, **kwargs)

            print('test', len(split_idx['test']))
            test_path = os.path.join(tempdir, 'samples_test')
            BulkSampleLoader(cugraph_store, cugraph_store,
                             input_nodes=split_idx['test'],
                             directory=test_path, **kwargs)

        dist.barrier()
    else:
        from torch_geometric.loader import NeighborLoader
        num_work = pyg_num_work(world_size)
        train_loader = NeighborLoader(data, input_nodes=split_idx['train'],
                                      num_workers=num_work, shuffle=True,
                                      drop_last=True, **kwargs)
        eval_loader = NeighborLoader(data, input_nodes=split_idx['valid'],
                                     num_workers=num_work, **kwargs)
        test_loader = NeighborLoader(data, input_nodes=split_idx['test'],
                                     num_workers=num_work, **kwargs)

    dist.barrier()
    eval_steps = 1000
    warmup_steps = 20
    acc = Accuracy(task="multiclass", num_classes=num_classes).to(rank)
    if rank == 0:
        print("Beginning training...")
    for epoch in range(epochs):
        if cugraph_data_loader:
            train_path = os.path.join(tempdir, f'samples_{epoch}')

            input_files = np.array_split(np.array(os.listdir(train_path)),
                                         world_size)[rank]

            train_loader = BulkSampleLoader(cugraph_store, cugraph_store,
                                            directory=train_path,
                                            input_files=input_files)
        with Join([model], divide_by_initial_world_size=False):
            for i, batch in enumerate(train_loader):
                if i == warmup_steps:
                    start = time.time()
                batch = batch.to(rank)

                if isinstance(batch, torch_geometric.data.HeteroData):
                    batch = batch.to_homogeneous()
                batch_size = batch.num_sampled_nodes[0]

                batch.y = batch.y.to(torch.long)
                optimizer.zero_grad()
                out = model(batch.x, batch.edge_index)
                loss = F.cross_entropy(out[:batch_size], batch.y[:batch_size])
                loss.backward()
                optimizer.step()
                if rank == 0 and i % 10 == 0:
                    print("Epoch: " + str(epoch) + ", Iteration: " + str(i) +
                          ", Loss: " + str(loss))
        dist.barrier()
        torch.cuda.synchronize()
        if cugraph_data_loader:
            eval_path = os.path.join(tempdir, f'samples_eval_{epoch}')

            input_files = np.array(os.listdir(eval_path))

            eval_loader = BulkSampleLoader(cugraph_store, cugraph_store,
                                           directory=eval_path,
                                           input_files=input_files)
        with Join([model], divide_by_initial_world_size=False):
            print("Average Training Iteration Time:",
                  (time.time() - start) / (i - warmup_steps), "s/iter")
            acc_sum = 0.0
            with torch.no_grad():
                for i, batch in enumerate(eval_loader):
                    if i >= eval_steps:
                        break

                    batch = batch.to(rank)
                    if isinstance(batch, torch_geometric.data.HeteroData):
                        batch = batch.to_homogeneous()
                    batch_size = batch.num_sampled_nodes[0]

                    batch.y = batch.y.to(torch.long)
                    out = model.module(batch.x, batch.edge_index)
                    acc_sum += acc(out[:batch_size].softmax(dim=-1),
                                   batch.y[:batch_size])

            acc_sum = torch.tensor(float(acc_sum), dtype=torch.float32,
                                   device=rank)
            dist.all_reduce(acc_sum, op=dist.ReduceOp.SUM)
            nb = torch.tensor(float(i), dtype=torch.float32,
                              device=acc_sum.device)
            dist.all_reduce(nb, op=dist.ReduceOp.SUM)
            print(f"Validation Accuracy: {acc_sum/(nb) * 100.0:.4f}%", )
        dist.barrier()

    with Join([model], divide_by_initial_world_size=False):
        if cugraph_data_loader:
            test_path = os.path.join(tempdir, 'samples_test')

            input_files = np.array(os.listdir(test_path))

            test_loader = BulkSampleLoader(cugraph_store, cugraph_store,
                                           directory=test_path,
                                           input_files=input_files)
        acc_sum = 0.0
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                batch = batch.to(rank)
                if isinstance(batch, torch_geometric.data.HeteroData):
                    batch = batch.to_homogeneous()
                batch_size = batch.num_sampled_nodes[0]

                batch.y = batch.y.to(torch.long)
                out = model.module(batch.x, batch.edge_index)
                acc_sum += acc(out[:batch_size].softmax(dim=-1),
                               batch.y[:batch_size])

            acc_sum = torch.tensor(float(acc_sum), dtype=torch.float32,
                                   device=rank)
            dist.all_reduce(acc_sum, op=dist.ReduceOp.SUM)
            nb = torch.tensor(float(i), dtype=torch.float32,
                              device=acc_sum.device)
            dist.all_reduce(nb, op=dist.ReduceOp.SUM)
            print(f"Test Accuracy: {acc_sum/(nb) * 100.0:.4f}%", )
    dist.barrier()

    if cugraph_data_loader and rank == 0:
        import gc
        del cugraph_store
        gc.collect()
        shutdown_dask_client(client)
    dist.barrier()
    if rank == 0:
        print("Total Program Runtime =",
              round(time.perf_counter() - wall_clock_start, 2), "seconds")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--fan_out', type=int, default=10)
    parser.add_argument(
        "--use_gat_conv",
        action='store_true',
        help="Whether or not to use GATConv. (Defaults to using GCNConv)",
    )
    parser.add_argument(
        "--n_gat_conv_heads",
        type=int,
        default=4,
        help="If using GATConv, number of attention heads to use",
    )
    parser.add_argument(
        "--cugraph_data_loader",
        action='store_true',
        help="Whether or not to use CuGraph for Neighbor Loading. \
            \nNote that this requires more GPU memory or \
            a reduction in batch_size/fan_out/hidden_channels/num_layers",
    )

    args = parser.parse_args()
    wall_clock_start = time.perf_counter()

    dataset = PygNodePropPredDataset(name='ogbn-papers100M',
                                     root='/datasets/ogb_datasets')
    split_idx = dataset.get_idx_split()
    data = dataset[0]
    data.y = data.y.reshape(-1)
    if args.use_gat_conv:
        model = torch_geometric.nn.models.GAT(dataset.num_features,
                                              args.hidden_channels,
                                              args.num_layers,
                                              dataset.num_classes,
                                              heads=args.n_gat_conv_heads)
    else:
        model = torch_geometric.nn.models.GCN(dataset.num_features,
                                              args.hidden_channels,
                                              args.num_layers,
                                              dataset.num_classes)

    print("Data =", data)
    world_size = torch.cuda.device_count()
    print('Let\'s use', world_size, 'GPUs!')

    with tempfile.TemporaryDirectory() as tempdir:
        mp.spawn(
            run_train, args=(data, world_size, model, args.epochs,
                             args.batch_size, args.fan_out, split_idx,
                             dataset.num_classes, args.cugraph_data_loader,
                             wall_clock_start, tempdir, args.num_layers),
            nprocs=world_size, join=True)
