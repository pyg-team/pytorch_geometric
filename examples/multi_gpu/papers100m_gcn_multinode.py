"""Multi-node multi-GPU example on ogbn-papers100m.

To run:
srun -l -N<num_nodes> --ntasks-per-node=<ngpu_per_node> \
--container-name=cont --container-image=<image_url> \
--container-mounts=/ogb-papers100m/:/workspace/dataset
python3 path_to_script.py
"""
import os
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from ogb.nodeproppred import PygNodePropPredDataset
from torch.nn.parallel import DistributedDataParallel

from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import GCNConv


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


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x


def run(world_size, data, split_idx, model):
    local_id = int(os.environ['LOCAL_RANK'])
    rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_id)
    device = torch.device(local_id)
    if rank == 0:
        print(f'Using {nprocs} GPUs...')

    split_idx['train'] = split_idx['train'].split(
        split_idx['train'].size(0) // world_size,
        dim=0,
    )[rank].clone()

    model = DistributedDataParallel(model.to(device), device_ids=[local_id])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    kwargs = dict(
        data=data,
        batch_size=128,
        num_workers=get_num_workers(),
        num_neighbors=[50, 50],
    )

    train_loader = NeighborLoader(
        input_nodes=split_idx['train'],
        shuffle=True,
        **kwargs,
    )
    if rank == 0:
        val_loader = NeighborLoader(input_nodes=split_idx['valid'], **kwargs)
        test_loader = NeighborLoader(input_nodes=split_idx['test'], **kwargs)

    val_steps = 1000
    warmup_steps = 100
    if rank == 0:
        print("Beginning training...")

    for epoch in range(1, 4):
        model.train()
        for i, batch in enumerate(train_loader):
            if i == warmup_steps:
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

        if rank == 0:
            sec_per_iter = (time.time() - start) / (i - warmup_steps)
            print(f"Avg Training Iteration Time: {sec_per_iter:.6f} s/iter")

            model.eval()
            total_correct = total_examples = 0
            for i, batch in enumerate(val_loader):
                if i >= val_steps:
                    break
                if i == warmup_steps:
                    start = time.time()

                batch = batch.to(device)
                with torch.no_grad():
                    out = model(batch.x, batch.edge_index)[:batch.batch_size]
                pred = out.argmax(dim=-1)
                y = batch.y[:batch.batch_size].view(-1).to(torch.long)

                total_correct += int((pred == y).sum())
                total_examples += y.size(0)

            print(f"Val Acc: {total_correct / total_examples:.4f}")
            sec_per_iter = (time.time() - start) / (i - warmup_steps)
            print(f"Avg Inference Iteration Time: {sec_per_iter:.6f} s/iter")

    if rank == 0:
        model.eval()
        total_correct = total_examples = 0
        for i, batch in enumerate(test_loader):
            batch = batch.to(device)
            with torch.no_grad():
                out = model(batch.x, batch.edge_index)[:batch.batch_size]
            pred = out.argmax(dim=-1)
            y = batch.y[:batch.batch_size].view(-1).to(torch.long)

            total_correct += int((pred == y).sum())
            total_examples += y.size(0)
        print(f"Test Acc: {total_correct / total_examples:.4f}")


if __name__ == '__main__':
    # Setup multi-node:
    torch.distributed.init_process_group("nccl")
    nprocs = dist.get_world_size()
    assert dist.is_initialized(), "Distributed cluster not initialized"
    dataset = PygNodePropPredDataset(name='ogbn-papers100M')
    split_idx = dataset.get_idx_split()
    model = GCN(dataset.num_features, 64, dataset.num_classes)

    run(nprocs, dataset[0], split_idx, model)
