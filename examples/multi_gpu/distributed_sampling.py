import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel
from torchmetrics import Accuracy

from torch_geometric.datasets import Reddit
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv


class SAGE(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int,
                 out_channels: int, num_layers: int = 2):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = x.relu_()
                x = F.dropout(x, p=0.5, training=self.training)
        return x


def eval(loader, model, acc, rank):
    acc_sum = 0.0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            batch_size = batch.batch_size
            out = model.module(batch.x, batch.edge_index.to(rank))
            acc_sum += acc(out[:batch_size].softmax(dim=-1),
                           batch.y[:batch_size])

        nb = i + 1.0
        return acc_sum / (nb) * 100.0


def run(rank, world_size, dataset, start):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)

    data = dataset[0]
    # Move to device for faster feature fetch.
    data = data.to(rank, 'x', 'y')

    # Split indices into `world_size` many chunks:
    train_idx = data.train_mask.nonzero(as_tuple=False).view(-1)
    train_idx = train_idx.split(train_idx.size(0) // world_size)[rank]
    val_idx = data.val_mask.nonzero(as_tuple=False).view(-1)
    val_idx = val_idx.split(val_idx.size(0) // world_size)[rank]
    test_idx = data.val_mask.nonzero(as_tuple=False).view(-1)
    test_idx = test_idx.split(test_idx.size(0) // world_size)[rank]
    kwargs = dict(batch_size=1024, num_workers=4, persistent_workers=True)
    train_loader = NeighborLoader(data, input_nodes=train_idx,
                                  num_neighbors=[25, 10], shuffle=True,
                                  drop_last=True, **kwargs)
    val_loader = NeighborLoader(data, input_nodes=val_idx,
                                num_neighbors=[25,
                                               10], drop_last=True, **kwargs)
    test_loader = NeighborLoader(data, input_nodes=test_idx,
                                 num_neighbors=[25,
                                                10], drop_last=True, **kwargs)

    torch.manual_seed(12345)
    acc = Accuracy(task="multiclass", num_classes=dataset.num_classes).to(rank)
    model = SAGE(dataset.num_features, 256, dataset.num_classes).to(rank)
    model = DistributedDataParallel(model, device_ids=[rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    start_time = True
    for epoch in range(1, 21):
        model.train()
        for batch in train_loader:
            if start_time:
                start_time = False
                since = time.time()
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index.to(rank))[:batch.batch_size]
            loss = F.cross_entropy(out, batch.y[:batch.batch_size])
            loss.backward()
            optimizer.step()

        dist.barrier()

        if rank == 0:
            print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')

        if epoch % 5 == 0:
            model.eval()
            acc1 = eval(train_loader, model, acc, rank)
            acc2 = eval(val_loader, model, acc, rank)
            acc3 = eval(test_loader, model, acc, rank)
            if world_size > 1:
                acc1 = torch.tensor(float(acc1), dtype=torch.float32,
                                    device=rank)
                acc2 = torch.tensor(float(acc2), dtype=torch.float32,
                                    device=rank)
                acc3 = torch.tensor(float(acc3), dtype=torch.float32,
                                    device=rank)
                dist.all_reduce(acc1, op=dist.ReduceOp.SUM)
                dist.all_reduce(acc2, op=dist.ReduceOp.SUM)
                dist.all_reduce(acc3, op=dist.ReduceOp.SUM)
                acc1 /= world_size
                acc2 /= world_size
                acc3 /= world_size

            if rank == 0:
                print(f'Train: {acc1:.4f}, Val: {acc2:.4f}, Test: {acc3:.4f}')

        dist.barrier()

    dist.destroy_process_group()
    if rank == 0:
        print("Time from first minibatch to done training=",
              round(time.time() - since, 2))
        print("Total program time (e2e_time) = ",
              round(time.time() - start, 2))


if __name__ == '__main__':
    start = time.time()
    dataset = Reddit('../../data/Reddit')

    world_size = torch.cuda.device_count()
    print('Let\'s use', world_size, 'GPUs!')
    mp.spawn(run, args=(world_size, dataset, start), nprocs=world_size,
             join=True)
