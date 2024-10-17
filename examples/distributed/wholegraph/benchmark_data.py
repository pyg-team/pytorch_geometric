"""Multi-node multi-GPU example on ogbn-papers100m.

Example way to run using srun:
srun -l -N<num_nodes> --ntasks-per-node=<ngpu_per_node> \
--container-name=cont --container-image=<image_url> \
--container-mounts=/ogb-papers100m/:/workspace/dataset
python3 path_to_script.py
"""
import os
import time
from typing import Optional
import argparse

import torch
import torch.distributed as dist
import torch.nn.functional as F
from ogb.nodeproppred import PygNodePropPredDataset
from torch.nn.parallel import DistributedDataParallel
from torchmetrics import Accuracy

from torch_geometric.loader import NodeLoader, NeighborLoader
from torch_geometric.nn import GCN

from torch_geometric.sampler import BaseSampler

from nv_distributed_graph import dist_shmem
from feature_store import WholeGraphFeatureStore
from graph_store import WholeGraphGraphStore

class WholeGraphSampler(BaseSampler):
    r"""
    A naive sampler class for WholeGraph graph storage that only supports uniform node-based sampling on homogeneous graph.
    """
    from torch_geometric.sampler import SamplerOutput, NodeSamplerInput
    def __init__(
        self,
        graph: WholeGraphGraphStore,
        num_neighbors,
    ):
        import pylibwholegraph.torch as wgth

        self.num_neighbors = num_neighbors
        self.wg_sampler = wgth.GraphStructure()
        row_indx, col_ptrs, _ = graph.csc()
        self.wg_sampler.set_csr_graph(col_ptrs._tensor, row_indx._tensor)

    def sample_from_nodes(
        self,
        inputs: NodeSamplerInput
    ) -> SamplerOutput:
        r"""
        Sample subgraphs from the given nodes based on uniform node-based sampling.
        """
        seed = inputs.node.cuda(non_blocking=True) # WholeGraph Sampler needs all seeds on device
        WG_SampleOutput = self.wg_sampler.multilayer_sample_without_replacement(seed, self.num_neighbors, None)
        out = WholeGraphGraphStore.create_pyg_subgraph(WG_SampleOutput)
        out.metadata = (inputs.input_id, inputs.time)
        return out

def run(world_size, rank, local_rank, device, mode):
    wall_clock_start = time.perf_counter()

    # Will query the runtime environment for `MASTER_ADDR` and `MASTER_PORT`.
    # Make sure, those are set!
    dist.init_process_group('nccl', world_size=world_size, rank=rank)
    dist_shmem.init_process_group_per_node()

    # Load the dataset in the local root process and share it with local ranks
    if dist_shmem.get_local_rank() == 0:
        dataset = PygNodePropPredDataset(name='ogbn-products', root='/workspace')
    else:
        dataset = None
    dataset = dist_shmem.to_shmem(dataset) # move dataset to shmem

    split_idx = dataset.get_idx_split()
    split_idx['train'] = split_idx['train'].split(
        split_idx['train'].size(0) // world_size, dim=0)[rank].clone()
    split_idx['valid'] = split_idx['valid'].split(
        split_idx['valid'].size(0) // world_size, dim=0)[rank].clone()
    split_idx['test'] = split_idx['test'].split(
        split_idx['test'].size(0) // world_size, dim=0)[rank].clone()
    data = dataset[0]
    num_features = dataset.num_features
    num_classes = dataset.num_classes

    if mode == 'baseline':
        data = data
        kwargs = dict(
            data=data,
            batch_size=1024,
            num_neighbors=[30, 30],
            num_workers=4,
        )
        train_loader = NeighborLoader(
            input_nodes=split_idx['train'],
            shuffle=True,
            drop_last=True,
            **kwargs,
        )
        val_loader = NeighborLoader(input_nodes=split_idx['valid'], **kwargs)
        test_loader = NeighborLoader(input_nodes=split_idx['test'], **kwargs)

    elif mode == 'UVA-features':
        feature_store = WholeGraphFeatureStore(pyg_data=data)
        graph_store = WholeGraphGraphStore(pyg_data=data, format='pyg')
        data = (feature_store, graph_store)
        kwargs = dict(
            data=data,
            batch_size=1024,
            num_neighbors=[30, 30],
            num_workers=4,
            filter_per_worker=False, # WholeGraph feature fetching is not fork-safe
        )
        train_loader = NeighborLoader(
            input_nodes=split_idx['train'],
            shuffle=True,
            drop_last=True,
            **kwargs,
        )
        val_loader = NeighborLoader(input_nodes=split_idx['valid'], **kwargs)
        test_loader = NeighborLoader(input_nodes=split_idx['test'], **kwargs)

    elif mode == 'UVA':
        feature_store = WholeGraphFeatureStore(pyg_data=data)
        graph_store = WholeGraphGraphStore(pyg_data=data)
        data = (feature_store, graph_store)
        kwargs = dict(
            data=data,
            batch_size=1024,
            num_workers=0, # with wholegraph sampler you don't need workers
            filter_per_worker=False, # WholeGraph feature fetching is not fork-safe
        )
        node_sampler = WholeGraphSampler(
            graph_store,
            num_neighbors=[30, 30],
        )
        train_loader = NodeLoader(
            input_nodes=split_idx['train'],
            node_sampler=node_sampler,
            shuffle=True,
            drop_last=True,
            **kwargs,
        )
        val_loader = NodeLoader(input_nodes=split_idx['valid'], node_sampler=node_sampler, **kwargs)
        test_loader = NodeLoader(input_nodes=split_idx['test'], node_sampler=node_sampler, **kwargs)

    eval_steps = 1000
    model = GCN(num_features, 256, 2, num_classes)
    acc = Accuracy(task="multiclass", num_classes=num_classes).to(device)
    model = DistributedDataParallel(model.to(device), device_ids=[local_rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001,
                                weight_decay=5e-4)

    if rank == 0:
        prep_time = round(time.perf_counter() - wall_clock_start, 2)
        print("Total time before training begins (prep_time)=", prep_time,
              "seconds")
        print("Beginning training...")

    for epoch in range(1, 21):
        dist.barrier()
        start = time.time()
        model.train()
        for i, batch in enumerate(train_loader):
            batch = batch.to(device)
            optimizer.zero_grad()
            y = batch.y[:batch.batch_size].view(-1).to(torch.long)
            out = model(batch.x, batch.edge_index)[:batch.batch_size]
            loss = F.cross_entropy(out, y)
            loss.backward()
            optimizer.step()
            if rank == 0 and i % 100 == 0:
                print(f'Epoch: {epoch:02d}, Iteration: {i}, Loss: {loss:.4f}')

        # Profile run:
        # We synchronize before barrier to flush GPU OPs first,
        # then adding barrier to sync CPUs to find max train time among all ranks.
        torch.cuda.synchronize()
        dist.barrier()
        epoch_end = time.time()

        @torch.no_grad()
        def test(loader: NodeLoader, num_steps: Optional[int] = None):
            model.eval()
            for j, batch in enumerate(loader):
                if num_steps is not None and j >= num_steps:
                    break
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index)[:batch.batch_size]
                y = batch.y[:batch.batch_size].view(-1).to(torch.long)
                acc(out, y)
            acc_sum = acc.compute()
            return acc_sum

        eval_acc = test(val_loader, num_steps=eval_steps)
        if rank == 0:
            print(f"Val Accuracy: {eval_acc:.4f}%", )
            print(
                f"Epoch {epoch:05d} | "
                f"Accuracy {eval_acc:.4f} | "
                f"Time {epoch_end - start:.2f}"
            )

        acc.reset()
        dist.barrier()

    test_acc = test(test_loader)
    if rank == 0:
        print(f"Test Accuracy: {test_acc:.4f}%", )
    dist.destroy_process_group() if dist.is_initialized() else None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='baseline', choices=['baseline', 'UVA-features', 'UVA'])
    args = parser.parse_args()

    # Get the world size from the WORLD_SIZE variable or directly from SLURM:
    world_size = int(
        os.environ.get('WORLD_SIZE', os.environ.get('SLURM_NTASKS')))
    # Likewise for RANK and LOCAL_RANK:
    rank = int(os.environ.get('RANK', os.environ.get('SLURM_PROCID')))
    local_rank = int(
        os.environ.get('LOCAL_RANK', os.environ.get('SLURM_LOCALID')))

    assert torch.cuda.is_available()
    device = torch.device(local_rank)
    torch.cuda.set_device(device)
    run(world_size, rank, local_rank, device, args.mode)

