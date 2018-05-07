import os.path as osp

import torch
import torch_geometric.transform as T
from torch_geometric.dataset import ENZYMES
from torch_geometric.data import DataLoader

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'ENZYMES2')
split = torch.randperm(ENZYMES.num_graphs)
dataset = ENZYMES(path, split, transform=T.TargetIndegree())
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

for data in dataloader:
    print(data.x.size(), data.edge_attr.size(), data.y.size())
