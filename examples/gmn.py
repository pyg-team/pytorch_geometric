import argparse
import os.path as osp

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Embedding, Linear, ModuleList, ReLU, Sequential
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch_geometric.data import DataLoader
from torch_geometric.datasets import ZINC
from torch_geometric.nn import BatchNorm, GMNConv, global_add_pool

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--early_stopping', type=int, default=10,
                    help='Tolerance for early stopping (# of epochs).')
parser.add_argument('--max_degree', type=int, default=3,
                    help='Maximum Chebyshev polynomial degree.')
parser.add_argument('--start_test', type=int, default=80,
                    help='define from which epoch test')
parser.add_argument('--train_jump', type=int, default=0,
                    help='define whether train jump, defaul train_jump=0')
parser.add_argument('--aggregators', type=str, default="mean,max,min",
                    help='choose your aggregators')

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"

device = torch.device(dev)
print("Device:", device)

args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'ZINC')
train_dataset = ZINC(path, subset=True, split='train')
val_dataset = ZINC(path, subset=True, split='val')
test_dataset = ZINC(path, subset=True, split='test')

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)
test_loader = DataLoader(test_dataset, batch_size=64)

deg = GMNConv.get_degree_histogram(train_loader)


class GraphMixerNet(torch.nn.Module):
    def __init__(self):
        super(GraphMixerNet, self).__init__()

        self.node_emb = Embedding(21, 75)
        self.edge_emb = Embedding(4, 50)

        aggregators = ['mean', 'min', 'max']
        scalers = ['identity', 'amplification', 'attenuation']

        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        for _ in range(4):

            conv = GMNConv(in_channels=75, out_channels=75,
                           aggregators=aggregators, scalers=scalers, deg=deg,
                           edge_dim=50, towers=5, post_layers=1,
                           divide_input=True)
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(75))

        self.mlp = Sequential(Linear(75, 50), ReLU(), Linear(50, 25), ReLU(),
                              Linear(25, 1))

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.node_emb(x.squeeze())
        edge_attr = self.edge_emb(edge_attr)
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = F.relu(batch_norm(conv(x, edge_index, edge_attr)))
        x = global_add_pool(x, batch)
        return self.mlp(x)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GraphMixerNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20,
                              min_lr=0.00001)


def train(epoch):
    model.train()

    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        loss = (out.squeeze() - data.y).abs().mean()
        loss.backward()
        total_loss += loss.item() * data.num_graphs
        optimizer.step()
    return total_loss / len(train_loader.dataset)


@torch.no_grad()
def test(loader):
    model.eval()

    total_error = 0
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        total_error += (out.squeeze() - data.y).abs().sum().item()
    return total_error / len(loader.dataset)


for epoch in range(args.epochs):
    loss = train(epoch)
    val_mae = test(val_loader)
    test_mae = test(test_loader)
    scheduler.step(val_mae)
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Val: {val_mae:.4f}, '
          f'Test: {test_mae:.4f}')
