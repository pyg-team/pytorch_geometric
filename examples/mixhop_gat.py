import os.path as osp

import numpy as np
import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.datasets import MixHopSyntheticDataset
from torch_geometric.data import DataLoader
import torch_geometric.transforms as T
from torch_geometric.nn import GATConv
from torch import optim

homophily="0.9"

path = osp.join('/home/laurent/graph_data', 'MixHop')
dataset = MixHopSyntheticDataset(path, homophily=homophily)
data = dataset[0]
train_loader = DataLoader(data.train_mask, batch_size=1, shuffle=True)
val_loader   = DataLoader(data.val_mask,   batch_size=1, shuffle=True)
test_loader  = DataLoader(data.test_mask,  batch_size=1, shuffle=True)

# the max of each feature is roughly 650
data.x = data.x / 650.

num_hidden = 8

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.lin = Linear(dataset.num_features, num_hidden)
        self.conv1 = GATConv(num_hidden, num_hidden, heads=2, dropout=0.)
        self.conv2 = GATConv(2 * num_hidden, num_hidden, heads=2, dropout=0.)
        self.conv3 = GATConv(2 * num_hidden, dataset.num_classes, heads=1, concat=False,
                             dropout=0.)
        
    def forward(self, x, edge_index):
        
        x = F.leaky_relu(self.lin(x))
        x = F.leaky_relu(self.conv1(x, edge_index))
        x = F.leaky_relu(self.conv2(x, edge_index))
        x = self.conv3(x, data.edge_index)
        return F.log_softmax(x, dim=-1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, data = Net().to(device), data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

def train(data):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()


def test(data):
    model.eval()
    out, accs = model(data.x, data.edge_index), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = out[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


for epoch in range(1, 201):
    train(data)
    train_acc, val_acc, test_acc = test(data)
    print(f'Epoch: {epoch:03d}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, '
          f'Test: {test_acc:.4f}')
