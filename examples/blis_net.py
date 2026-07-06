"""BLIS-Net example on graph classification.

BLIS-Net is primarily designed for signal-level classification tasks (i.e.
predicting a label for each signal defined on a fixed graph). This example
instead runs graph classification on MUTAG simply because it is a readily
available dataset in PyG; the same :class:`~torch_geometric.nn.BLISConv`
scattering layers apply directly to signal-level tasks.

The model stacks :class:`~torch_geometric.nn.BLISConv` bi-Lipschitz
geometric-scattering layers into a cascade, reads out a graph-level
representation with a (first-order moment) mean pooling, and classifies with an
MLP. This mirrors the BLIS-Net architecture from the `"BLIS-Net: Classifying
and Analyzing Signals on Graphs"
<https://proceedings.mlr.press/v238/xu24c.html>`_ paper: since the scattering
layers preserve the full wavelet-frame energy, only the final layer's
coefficients are pooled.
"""
import argparse
import os.path as osp

import torch
import torch.nn.functional as F

from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import BLISConv, global_mean_pool

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='MUTAG')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--hidden_channels', type=int, default=64)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--epochs', type=int, default=100)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'TU')
dataset = TUDataset(path, name=args.dataset).shuffle()

n = (len(dataset) + 9) // 10
test_dataset = dataset[:n]
train_dataset = dataset[n:]
train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, args.batch_size)


class BLISNet(torch.nn.Module):
    def __init__(self, in_channels, num_classes, num_layers, hidden_channels):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        channels = in_channels
        for _ in range(num_layers):
            conv = BLISConv(channels)
            self.convs.append(conv)
            channels = conv.out_channels

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, num_classes),
        )

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index)  # Final-layer readout (energy preserving).
        x = global_mean_pool(x, batch)  # First-order moment aggregation.
        return self.mlp(x)


model = BLISNet(
    dataset.num_features,
    dataset.num_classes,
    args.num_layers,
    args.hidden_channels,
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.01)


def train():
    model.train()
    total_loss = 0.0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = F.cross_entropy(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(train_loader.dataset)


@torch.no_grad()
def test(loader):
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        pred = model(data.x, data.edge_index, data.batch).argmax(dim=-1)
        correct += int((pred == data.y).sum())
    return correct / len(loader.dataset)


for epoch in range(1, args.epochs + 1):
    loss = train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, '
          f'Train: {train_acc:.4f}, Test: {test_acc:.4f}')
