import os.path as osp

import torch_geometric.transforms as T
from torch_geometric.dataset import ModelNet10
from torch_geometric.data import DataLoader

path = osp.join(osp.dirname(osp.realpath(__file__)), '../data/ModelNet10')
dataset = ModelNet10(path, train=True, transform=T.Cartesian())
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

for data in dataloader:
    print(data.edge_index.size(), data.y.size())
