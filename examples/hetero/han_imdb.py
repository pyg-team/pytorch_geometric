# TODO: use pathlib instead of os.path
# add linear layer after han
# add patience
# add multiple heads
# add dropout

from typing import Union, Dict
import os.path as osp
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric import datasets

from torch_geometric.datasets import IMDB
from torch_geometric.nn import HANConv
import torch_geometric.transforms as T

path = osp.join(osp.dirname(osp.realpath(__file__)), '../../data/IMDB')
metapaths = [[('movie','actor'),('actor','movie')],
             [('movie','director'),('director','movie')]]
transform = T.AddMetaPaths(metapaths=metapaths, drop_orig_edges=True)
dataset = IMDB(path, transform=transform)
data = dataset[0]
del data['director'], data['actor']

class han(nn.Module):
    def __init__(self, in_channels: Union[int, Dict[str,int]], hidden_channels=128,
                 heads = 8, out_channels = 4):
        super().__init__()
        self.han_conv = HANConv(in_channels=in_channels,out_channels=hidden_channels, heads = heads,
                           metadata= data.metadata())
        self.lin = nn.Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        
        out = self.han_conv(x_dict, edge_index_dict)
        out = self.lin(out['movie'])
        return out

model = han(in_channels=-1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data, model = data.to(device), model.to(device)

with torch.no_grad():  # Initialize lazy modules.
    out = model(data.x_dict, data.edge_index_dict)

optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.001)


def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x_dict, data.edge_index_dict)
    mask = data['movie'].train_mask
    loss = F.cross_entropy(out[mask], data['movie'].y[mask])
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test():
    model.eval()
    pred = model(data.x_dict, data.edge_index_dict).argmax(dim=-1)

    accs = []
    for split in ['train_mask', 'val_mask', 'test_mask']:
        mask = data['movie'][split]
        acc = (pred[mask] == data['movie'].y[mask]).sum() / mask.sum()
        accs.append(float(acc))
    return accs


for epoch in range(1, 10):
    loss = train()
    train_acc, val_acc, test_acc = test()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, '
          f'Val: {val_acc:.4f}, Test: {test_acc:.4f}')