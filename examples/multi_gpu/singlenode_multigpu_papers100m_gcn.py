import argparse
import os

import numpy as np

os.environ['CUDF_SPILL'] = '1'
os.environ['RAPIDS_NO_INITIALIZE'] = '1'


def start_dask_cluster():
    from cugraph.testing.mg_utils import enable_spilling
    from dask_cuda import LocalCUDACluster

    cluster = LocalCUDACluster(
        protocol="tcp",
        rmm_pool_size=None,
        memory_limit=None,
    )

    from dask.distributed import Client
    client = Client(cluster)
    client.wait_for_workers(n_workers=len(cluster.workers))
    client.run(enable_spilling)

    print("Dask Cluster Setup Complete")
    del client
    return cluster


def create_dask_client(scheduler_address):
    from cugraph.dask.comms import comms as Comms
    from dask.distributed import Client, Lock

    client = Client(scheduler_address)
    lock = Lock('comms_init')
    if lock.acquire(timeout=100):
        try:
            Comms.initialize(p2p=True)
        finally:
            lock.release()
    else:
        raise RuntimeError("Failed to acquire lock to initialize comms")

    return client


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
        import cupy
        import rmm
        import torch
        """
        rmm.reinitialize(
            devices=[rank],
            pool_allocator=False,
            managed_memory=False
        )
        """

        #if rank == 0:
        #    from rmm.allocators.torch import rmm_torch_allocator
        #    torch.cuda.memory.change_current_allocator(rmm_torch_allocator)

        cupy.cuda.Device(rank).use()
        from rmm.allocators.cupy import rmm_cupy_allocator
        cupy.cuda.set_allocator(rmm_cupy_allocator)

        from cugraph.testing.mg_utils import enable_spilling
        enable_spilling()

        torch.cuda.set_device(rank)

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    import torch.distributed as dist
    dist.init_process_group('nccl', rank=rank, world_size=world_size)


def run_train(rank, data, world_size, model, epochs, batch_size, fan_out,
              split_idx, num_classes, cugraph_data_loader,
              scheduler_address=None, tempdir=None):
    import time

    import torch
    import torch.nn.functional as F
    from torch.nn.parallel import DistributedDataParallel
    from torchmetrics import Accuracy

    init_pytorch_worker(
        rank,
        world_size,
        cugraph_data_loader=cugraph_data_loader,
    )

    if cugraph_data_loader:
        print(f'creating dask client on rank {rank}')
        client = create_dask_client(scheduler_address)
        print(f'created dask client on rank {rank}')
    else:
        split_idx['train'] = split_idx['train'].split(
            split_idx['train'].size(0) // world_size, dim=0)[rank].clone()
    model = model.to(rank)
    model = DistributedDataParallel(model, device_ids=[rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01,
                                 weight_decay=0.0005)

    kwargs = dict(
        num_neighbors=[fan_out, fan_out],
        batch_size=batch_size,
    )
    # Set Up Neighbor Loading
    if cugraph_data_loader:
        import cugraph
        from cugraph_pyg.data import CuGraphStore
        from cugraph_pyg.loader import BulkSampleLoader, CuGraphNeighborLoader
        G = {("N", "E", "N"): data.edge_index}
        N = {"N": data.num_nodes}
        fs = cugraph.gnn.FeatureStore(backend="torch")
        fs.add_data(data.x, "N", "x")
        fs.add_data(data.y, "N", "y")

        from distributed import Event as Dask_Event
        event = Dask_Event("cugraph_store_creation_event")

        import torch.distributed as dist
        from torch.distributed.algorithms.join import Join

        import torch_geometric
        dist.barrier()

        if rank == 0:
            print(
                "Rank 0 creating its cugraph store and initializing distributed graph"
            )
            cugraph_store = CuGraphStore(fs, G, N, multi_gpu=True)
            event.set()
            print("Distributed graph initialization complete.")
        else:
            print(f"Rank {rank} waiting for distributed graph initialization")
            if event.wait(timeout=1000):
                print(f"Rank {rank} proceeding with store creation")
                cugraph_store = CuGraphStore(fs, {
                    k: len(v)
                    for k, v in G.items()
                }, N, multi_gpu=False)
                print(f"Rank {rank} created store")

        dist.barrier()
        if rank == 0:
            for epoch in range(epochs):
                train_path = os.path.join(tempdir, f'samples_{epoch}')
                os.mkdir(train_path)
                # runs sampling for the training epoch
                BulkSampleLoader(
                    cugraph_store,
                    cugraph_store,
                    input_nodes=split_idx['train'],
                    directory=train_path,
                    #shuffle=True, drop_last=True,
                    **kwargs)

            print('validation', len(split_idx['valid']))
            eval_loader = CuGraphNeighborLoader(cugraph_store,
                                                input_nodes=split_idx['valid'],
                                                **kwargs)
            test_loader = CuGraphNeighborLoader(cugraph_store,
                                                input_nodes=split_idx['test'],
                                                **kwargs)

        dist.barrier()
    else:
        from torch_geometric.loader import NeighborLoader
        num_work = pyg_num_work(world_size)
        train_loader = NeighborLoader(data, input_nodes=split_idx['train'],
                                      num_workers=num_work, shuffle=True,
                                      drop_last=True, **kwargs)

        if rank == 0:
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
        with Join([model]):
            for i, batch in enumerate(train_loader):
                if i >= warmup_steps:
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
        with Join([model]):
            if rank == 0:
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
                print(f"Validation Accuracy: {acc_sum/(i) * 100.0:.4f}%", )
        dist.barrier()

    with Join([model]):
        if rank == 0:
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
                print(f"Test Accuracy: {acc_sum/(i) * 100.0:.4f}%", )
    dist.barrier()

    if cugraph_data_loader:
        shutdown_dask_client(client)
    dist.barrier()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_channels', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--fan_out', type=int, default=16)
    parser.add_argument(
        "--use_gat_conv",
        action='store_true',
        help="Wether or not to use GATConv. (Defaults to using GCNConv)",
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
        help="Wether or not to use CuGraph for Neighbor Loading. \
            \nNote that this requires more GPU memory or \
            a reduction in batch_size/fan_out/hidden_channels/num_layers",
    )

    args = parser.parse_args()

    cluster = start_dask_cluster() if args.cugraph_data_loader else None

    import torch
    import torch.multiprocessing as mp
    from ogb.nodeproppred import PygNodePropPredDataset

    import torch_geometric

    dataset = PygNodePropPredDataset(name='ogbn-papers100M')
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

    import tempfile
    with tempfile.TemporaryDirectory() as tempdir:
        mp.spawn(
            run_train,
            args=(data, world_size, model, args.epochs, args.batch_size,
                  args.fan_out, split_idx, dataset.num_classes,
                  args.cugraph_data_loader,
                  None if cluster is None else cluster.scheduler_address,
                  tempdir), nprocs=world_size, join=True)

    if cluster is not None:
        cluster.close()
