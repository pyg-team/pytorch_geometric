import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from ogb.nodeproppred import PygNodePropPredDataset
from torch.nn.parallel import DistributedDataParallel

from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import GCNConv


def get_num_workers(world_size: int) -> int:
    num_workers = None
    if hasattr(os, "sched_getaffinity"):
        try:
            num_workers = len(os.sched_getaffinity(0)) // (2 * world_size)
        except Exception:
            pass
    if num_workers is None:
        num_workers = os.cpu_count() // (2 * world_size)
    return num_workers


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index=None):
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x


def run(rank, world_size, data, split_idx, model):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)

    split_idx['train'] = split_idx['train'].split(
        split_idx['train'].size(0) // world_size,
        dim=0,
    )[rank].clone()

    model = DistributedDataParallel(model.to(rank), device_ids=[rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    kwargs = dict(
        data=data,
        batch_size=128,
        num_workers=get_num_workers(world_size),
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
            batch = batch.to(rank)
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

                batch = batch.to(rank)
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
            batch = batch.to(rank)
            with torch.no_grad():
                out = model(batch.x, batch.edge_index)[:batch.batch_size]
            pred = out.argmax(dim=-1)
            y = batch.y[:batch.batch_size].view(-1).to(torch.long)

            total_correct += int((pred == y).sum())
            total_examples += y.size(0)
        print(f"Test Acc: {total_correct / total_examples:.4f}")


if __name__ == '__main__':
    dataset = PygNodePropPredDataset(name='ogbn-papers100M')
    split_idx = dataset.get_idx_split()
    model = GCN(dataset.num_features, 64, dataset.num_classes)

    world_size = torch.cuda.device_count()
    print('Let\'s use', world_size, 'GPUs!')
    mp.spawn(
        run,
        args=(world_size, dataset[0], split_idx, model),
        nprocs=world_size,
        join=True,
    )
