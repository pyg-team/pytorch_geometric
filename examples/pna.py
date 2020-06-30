import os.path as osp
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import PNAConv

parser = argparse.ArgumentParser()
parser.add_argument('--aggregators', type=str, help='Aggregators to use.', default='mean max min std')
parser.add_argument('--scalers', type=str, help='Scalers to use.', default='identity amplification attenuation')
parser.add_argument('--towers', type=int, help='Towers to use.', default=5)
parser.add_argument('--divide_input', type=bool, help='Whether to divide the input between the towers.', default=False)
parser.add_argument('--gru', type=bool, help='Whether to use gru.', default=False)
parser.add_argument('--pretrans_layers', type=int, help='pretrans_layers.', default=1)
parser.add_argument('--posttrans_layers', type=int, help='posttrans_layers.', default=1)
args = parser.parse_args()

dataset = 'Cora'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
data = dataset[0]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device', device)
model = PNAConv(aggregators=args.aggregators.split(), scalers=args.scalers.split(), towers=args.towers,
                divide_input=args.divide_input,
                pretrans_layers=args.pretrans_layers, posttrans_layers=args.posttrans_layers,
                in_channels=dataset.num_node_features,
                out_channels=dataset.num_classes, avg_d=data.num_edges / data.num_nodes,
                edge_features=bool(dataset.num_edge_features)).to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train():
    model.train()
    optimizer.zero_grad()
    y = model(data.x, data.edge_index, data.edge_attr)[data.train_mask]
    train_loss, val_loss = [torch.nn.CrossEntropyLoss()(y, data.y[data.train_mask]) for mask in
                            (data.train_mask, data.val_mask)]
    train_loss.backward()
    optimizer.step()
    return train_loss.item(), val_loss.item()


@torch.no_grad()
def test():
    model.eval()
    y = model(data.x, data.edge_index, data.edge_attr)
    train_acc, val_acc, test_acc = [torch.mean((torch.argmax(y[mask], dim=1) == data.y[mask]).float()).item() for mask
                                    in (data.train_mask, data.val_mask, data.test_mask)]
    return train_acc, val_acc, test_acc


for epoch in range(1, 301):
    train_loss, val_loss = train()
    print('Epoch: {:03d}, Loss: {:.8f}, Val Loss {:.8f}'.format(epoch, train_loss, val_loss))

train_acc, val_acc, test_acc = test()
print('Accuracy: train: {:.4f} val: {:.4f} test: {:.4f}'.format(train_acc, val_acc, test_acc))
