import argparse
import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from ogb.nodeproppred import PygNodePropPredDataset
from torch.nn.parallel import DistributedDataParallel
from torchmetrics import Accuracy

from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import GCNConv


def pyg_num_work():
    num_work = None
    if hasattr(os, "sched_getaffinity"):
        try:
            num_work = len(os.sched_getaffinity(0)) / 2
        except Exception:
            pass
    if num_work is None:
        num_work = os.cpu_count() / 2
    return int(num_work)


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_weight=None):
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv1(x, edge_index, edge_weight).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x


def run_train(rank, data, world_size, model, epochs, batch_size, fan_out,
              split_idx, num_classes):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    split_idx['train'] = split_idx['train'].split(
        split_idx['train'].size(0) // world_size, dim=0)[rank].clone()
    model = model.to(rank)
    model = DistributedDataParallel(model, device_ids=[rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01,
                                 weight_decay=0.0005)
    train_loader = NeighborLoader(data, num_neighbors=[fan_out, fan_out],
                                  input_nodes=split_idx['train'],
                                  batch_size=batch_size,
                                  num_workers=pyg_num_work())
    if rank == 0:
        eval_loader = NeighborLoader(data, num_neighbors=[fan_out, fan_out],
                                     input_nodes=split_idx['valid'],
                                     batch_size=batch_size,
                                     num_workers=pyg_num_work())
        test_loader = NeighborLoader(data, num_neighbors=[fan_out, fan_out],
                                     input_nodes=split_idx['test'],
                                     batch_size=batch_size,
                                     num_workers=pyg_num_work())
    eval_steps = 100
    acc = Accuracy(task="multiclass", num_classes=num_classes).to(rank)
    if rank == 0:
        print("Beginning training...")
    for epoch in range(epochs):
        for i, batch in enumerate(train_loader):
            if i >= 10:
                start = time.time()
            batch = batch.to(rank)
            batch.y = batch.y.to(torch.long)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)
            loss = F.cross_entropy(out[:batch_size], batch.y[:batch_size])
            loss.backward()
            optimizer.step()
            if rank == 0 and i % 10 == 0:
                print("Epoch: " + str(epoch) + ", Iteration: " + str(i) +
                      ", Loss: " + str(loss))
        if rank == 0:
            print("Average Training Iteration Time:",
                  (time.time() - start) / (i - 10), "s/iter")
            acc_sum = 0.0
            with torch.no_grad():
                for i, batch in enumerate(eval_loader):
                    if i >= eval_steps:
                        break
                    if i >= 10:
                        start = time.time()
                    batch = batch.to(rank)
                    batch.y = batch.y.to(torch.long)
                    out = model(batch.x, batch.edge_index)
                    acc_sum += acc(out[:batch_size].softmax(dim=-1),
                                   batch.y[:batch_size])
            print(f"Validation Accuracy: {acc_sum/(i) * 100.0:.4f}%", )
            print("Average Inference Iteration Time:",
                  (time.time() - start) / (i - 10), "s/iter")
    if rank == 0:
        acc_sum = 0.0
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                batch = batch.to(rank)
                batch.y = batch.y.to(torch.long)
                out = model(batch.x, batch.edge_index)
                acc_sum += acc(out[:batch_size].softmax(dim=-1),
                               batch.y[:batch_size])
            print(f"Test Accuracy: {acc_sum/(i) * 100.0:.4f}%", )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--fan_out', type=int, default=50)

    args = parser.parse_args()

    dataset = PygNodePropPredDataset(name='ogbn-papers100M')
    split_idx = dataset.get_idx_split()
    data = dataset[0]
    data.y = data.y.reshape(-1)
    model = GCN(dataset.num_features, args.hidden_channels,
                dataset.num_classes)
    print("Data =", data)
    world_size = torch.cuda.device_count()
    print('Let\'s use', world_size, 'GPUs!')
    mp.spawn(
        run_train, args=(data, world_size, model, args.epochs, args.batch_size,
                         args.fan_out, split_idx, dataset.num_classes),
        nprocs=world_size, join=True)
