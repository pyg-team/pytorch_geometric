import os.path as osp

import torch
from torch_geometric.dataset import QM9
from torch_geometric.data import DataLoader

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'QM9')
split = torch.randperm(QM9.num_graphs)
dataset = QM9(path, split)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

for data in dataloader:
    print(data.x.size(), data.edge_index.size(), data.y.size())
