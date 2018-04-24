import os.path as osp

import torch
from torch_geometric.dataset import ENZYMES
from torch_geometric.data import DataLoader

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'ENZYMES')
split = torch.randperm(ENZYMES.num_graphs)
dataset = ENZYMES(path, split)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

for data in dataloader:
    print(data.x.size(), data.edge_index.size(), data.y.size())
