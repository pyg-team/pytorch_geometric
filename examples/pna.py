import os.path as osp
import argparse

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PNAConv(aggregators=args.aggregators.split(), scalers=args.scalers.split(), towers=args.towers,
                divide_input=args.divide_input,
                pretrans_layers=args.pretrans_layers, posttrans_layers=args.posttrans_layers, in_channels=dataset[0].x.shape[-1],
                out_channels=5, avg_d=3.14, edge_features=dataset[0].edge_attr is not None).to(device)
data = dataset[0].to(device)
print(data)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train():
    model.train()
    optimizer.zero_grad()
    pos_z, neg_z, summary = model(data.x, data.edge_index, data.edge_attr)
    loss = model.loss(pos_z, neg_z, summary)
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def test():
    model.eval()
    z, _, _ = model(data.x, data.edge_index)
    acc = model.test(z[data.train_mask], data.y[data.train_mask],
                     z[data.test_mask], data.y[data.test_mask], max_iter=150)
    return acc


for epoch in range(1, 301):
    loss = train()
    print('Epoch: {:03d}, Loss: {:.4f}'.format(epoch, loss))
acc = test()
print('Accuracy: {:.4f}'.format(acc))
