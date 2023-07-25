# Implementation of:
# Graph-less Neural Networks: Teaching Old MLPs New Tricks via Distillation

import argparse
import os.path as osp

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCN, MLP

parser = argparse.ArgumentParser()
parser.add_argument('--lamb', type=float, default=0.0,
                    help='Balances loss from hard labels and teacher outputs')
args = parser.parse_args()

if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Planetoid')
dataset = Planetoid(path, name='Cora', transform=T.NormalizeFeatures())
data = dataset[0].to(device)

gnn = GCN(dataset.num_node_features, hidden_channels=16,
          out_channels=dataset.num_classes, num_layers=2).to(device)
mlp = MLP([dataset.num_node_features, 64, dataset.num_classes], dropout=0.5,
          norm=None).to(device)

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

with torch.no_grad():  # Obtain soft labels from the GNN:
    y_soft = gnn(data.x, data.edge_index).log_softmax(dim=-1)


def train_student():
    mlp.train()
    mlp_optimizer.zero_grad()
    out = mlp(data.x)
    loss1 = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss2 = F.kl_div(out.log_softmax(dim=-1), y_soft, reduction='batchmean',
                     log_target=True)
    loss = args.lamb * loss1 + (1 - args.lamb) * loss2
    loss.backward()
    mlp_optimizer.step()
    return float(loss)


@torch.no_grad()
def test_student():
    mlp.eval()
    pred = mlp(data.x).argmax(dim=-1)
    accs = []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))
    return accs


print('Training Student MLP:')
for epoch in range(1, 501):
    loss = train_student()
    if epoch % 20 == 0:
        train_acc, val_acc, test_acc = test_student()
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, '
              f'Val: {val_acc:.4f}, Test: {test_acc:.4f}')
