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

import torch
import torch.distributed as dist
import torch.nn.functional as F
from ogb.nodeproppred import PygNodePropPredDataset
from torch.nn.parallel import DistributedDataParallel
from torchmetrics import Accuracy

from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import GCN


def get_num_workers() -> int:
    num_workers = None
    if hasattr(os, "sched_getaffinity"):
        try:
            num_workers = len(os.sched_getaffinity(0)) // 2
        except Exception:
            pass
    if num_workers is None:
        num_workers = os.cpu_count() // 2
    return num_workers


def run(world_size, data, split_idx, model, acc, wall_clock_start):
    local_id = int(os.environ['LOCAL_RANK'])
    rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_id)
    device = torch.device(local_id)
    if rank == 0:
        print(f'Using {nprocs} GPUs...')

    split_idx['train'] = split_idx['train'].split(
        split_idx['train'].size(0) // world_size, dim=0)[rank].clone()
    split_idx['valid'] = split_idx['valid'].split(
        split_idx['valid'].size(0) // world_size, dim=0)[rank].clone()
    split_idx['test'] = split_idx['test'].split(
        split_idx['test'].size(0) // world_size, dim=0)[rank].clone()

    model = DistributedDataParallel(model.to(device), device_ids=[local_id])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001,
                                 weight_decay=5e-4)

    kwargs = dict(
        data=data,
        batch_size=1024,
        num_workers=get_num_workers(),
        num_neighbors=[30, 30],
    )

    train_loader = NeighborLoader(
        input_nodes=split_idx['train'],
        shuffle=True,
        drop_last=True,
        **kwargs,
    )
    val_loader = NeighborLoader(input_nodes=split_idx['valid'], **kwargs)
    test_loader = NeighborLoader(input_nodes=split_idx['test'], **kwargs)

    val_steps = 1000
    warmup_steps = 100
    acc = acc.to(device)
    dist.barrier()
    torch.cuda.synchronize()
    if rank == 0:
        prep_time = round(time.perf_counter() - wall_clock_start, 2)
        print("Total time before training begins (prep_time)=", prep_time,
              "seconds")
        print("Beginning training...")

    for epoch in range(1, 21):
        model.train()
        for i, batch in enumerate(train_loader):
            if i == warmup_steps:
                torch.cuda.synchronize()
                start = time.time()
            batch = batch.to(device)
            optimizer.zero_grad()
            y = batch.y[:batch.batch_size].view(-1).to(torch.long)
            out = model(batch.x, batch.edge_index)[:batch.batch_size]
            loss = F.cross_entropy(out, y)
            loss.backward()
            optimizer.step()

            if rank == 0 and i % 10 == 0:
                print(f'Epoch: {epoch:02d}, Iteration: {i}, Loss: {loss:.4f}')

        dist.barrier()
        torch.cuda.synchronize()
        if rank == 0:
            sec_per_iter = (time.time() - start) / (i + 1 - warmup_steps)
            print(f"Avg Training Iteration Time: {sec_per_iter:.6f} s/iter")

        @torch.no_grad()
        def test(loader: NeighborLoader, num_steps: Optional[int] = None):
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

        eval_acc = test(val_loader, num_steps=val_steps)
        if rank == 0:
            print(f"Val Accuracy: {eval_acc:.4f}%", )

        acc.reset()
        dist.barrier()

    test_acc = test(test_loader)
    if rank == 0:
        print(f"Test Accuracy: {test_acc:.4f}%", )

    dist.barrier()
    acc.reset()
    torch.cuda.synchronize()

    if rank == 0:
        total_time = round(time.perf_counter() - wall_clock_start, 2)
        print("Total Program Runtime (total_time) =", total_time, "seconds")
        print("total_time - prep_time =", total_time - prep_time, "seconds")


if __name__ == '__main__':
    wall_clock_start = time.perf_counter()
    # Setup multi-node:
    torch.distributed.init_process_group("nccl")
    nprocs = dist.get_world_size()
    assert dist.is_initialized(), "Distributed cluster not initialized"
    dataset = PygNodePropPredDataset(name='ogbn-papers100M')
    split_idx = dataset.get_idx_split()
    model = GCN(dataset.num_features, 256, 2, dataset.num_classes)
    acc = Accuracy(task="multiclass", num_classes=dataset.num_classes)
    data = dataset[0]
    data.y = data.y.reshape(-1)
    run(nprocs, data, split_idx, model, acc, wall_clock_start)
