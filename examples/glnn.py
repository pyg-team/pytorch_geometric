# Implementation of:
# Graph-less Neural Networks: Teaching Old MLPs New Tricks via Distillation

import argparse
import os.path as osp

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import GCN, MLP
from torch_geometric.datasets import Planetoid

parser = argparse.ArgumentParser()
parser.add_argument('--lambda', type=float, default=0.5,
                    help='Balances loss from hard labels and teacher outputs')
args = parser.parse_args()

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Planetoid')
dataset = Planetoid(path, name='Cora', transform=T.NormalizeFeatures())
data = dataset[0]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gnn = GCN(dataset.num_node_features, hidden_channels=16,
          out_channels=dataset.num_classes, num_layers=2).to(device)
mlp = MLP([dataset.num_node_features, 16, dataset.num_classes], dropout=0.5,
          batch_norm=False).to(device)

gnn_optimizer = torch.optim.Adam(gnn.parameters(), lr=0.01, weight_decay=5e-4)
mlp_optimizer = torch.optim.Adam(mlp.parameters(), lr=0.01, weight_decay=5e-4)


def train_teacher():
    gnn.train()
    gnn_optimizer.zero_grad()
    out = gnn(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    gnn_optimizer.step()
    return float(loss)


@torch.no_grad()
def test_teacher():
    gnn.eval()
    pred = gnn(data.x, data.edge_index).argmax(dim=-1)
    accs = []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))
    return accs


print('Training Teacher GNN:')
for epoch in range(1, 201):
    loss = train_teacher()
    if epoch % 20 == 0:
        train_acc, val_acc, test_acc = test_teacher()
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, '
              f'Val: {val_acc:.4f}, Test: {test_acc:.4f}')
