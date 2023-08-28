import argparse
import os
import time

import torch
import torch.nn.functional as F
from ogb.nodeproppred import PygNodePropPredDataset
from torchmetrics import Accuracy

from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import GCNConv

parser = argparse.ArgumentParser()
parser.add_argument('--hidden_channels', type=int, default=64)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--epochs', type=int, default=3)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--fan_out', type=int, default=50)

args = parser.parse_args()

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

dataset = PygNodePropPredDataset(name='ogbn-papers100M')
split_idx = dataset.get_idx_split()
data = dataset[0]
data.y = data.y.reshape(-1)
print("Data =", data)


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


model = GCN(dataset.num_features, args.hidden_channels,
            dataset.num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0005)
batch_size = args.batch_size
train_loader = NeighborLoader(data, num_neighbors=[args.fan_out, args.fan_out],
                              input_nodes=split_idx['train'],
                              batch_size=batch_size,
                              num_workers=pyg_num_work())
eval_loader = NeighborLoader(data, num_neighbors=[args.fan_out, args.fan_out],
                             input_nodes=split_idx['valid'],
                             batch_size=batch_size, num_workers=pyg_num_work())
test_loader = NeighborLoader(data, num_neighbors=[args.fan_out, args.fan_out],
                             input_nodes=split_idx['test'],
                             batch_size=batch_size, num_workers=pyg_num_work())
eval_steps = 100
acc = Accuracy(task="multiclass", num_classes=dataset.num_classes).to(device)
for epoch in range(args.epochs):
    for i, batch in enumerate(train_loader):
        if i >= 10:
            start = time.time()
        batch = batch.to(device)
        batch.y = batch.y.to(torch.long)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index)
        loss = F.cross_entropy(out[:batch_size], batch.y[:batch_size])
        loss.backward()
        optimizer.step()
        if i % 10 == 0:
            print("Epoch: " + str(epoch) + ", Iteration: " + str(i) +
                  ", Loss: " + str(loss))
    print("Average Iteration Time:", (time.time() - start) / (i - 10),
          "s/iter")
    acc_sum = 0.0
    with torch.no_grad():
        for i, batch in enumerate(eval_loader):
            if i >= eval_steps:
                break
            batch = batch.to(device)
            batch.y = batch.y.to(torch.long)
            out = model(batch.x, batch.edge_index)
            acc_sum += acc(out[:batch_size].softmax(dim=-1),
                           batch.y[:batch_size])
        print(f"Validation Accuracy: {acc_sum/(i) * 100.0:.4f}%", )
acc_sum = 0.0
with torch.no_grad():
    for i, batch in enumerate(test_loader):
        batch = batch.to(device)
        batch.y = batch.y.to(torch.long)
        out = model(batch.x, batch.edge_index)
        acc_sum += acc(out[:batch_size].softmax(dim=-1), batch.y[:batch_size])
    print(f"Test Accuracy: {acc_sum/(i) * 100.0:.4f}%", )
