import os.path as osp

from torch_geometric.dataset import ModelNet10
from torch_geometric.data import DataLoader

path = osp.join(osp.dirname(osp.realpath(__file__)), '../data/ModelNet10')
dataset = ModelNet10(path, train=True)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

for data in dataloader:
    print(data.edge_index.size(), data.y.size())
