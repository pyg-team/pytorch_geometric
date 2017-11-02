from __future__ import division, print_function

import sys
from torchvision.transforms import Compose

sys.path.insert(0, '.')
sys.path.insert(0, '..')

from torch_geometric.datasets import MNISTSuperpixel75  # noqa
from torch_geometric.transform import Graclus, PolarAdj  # noqa
from torch_geometric.utils import DataLoader  # noqa

path = '~/MNISTSuperpixel75'
train_dataset = MNISTSuperpixel75(
    path, train=True, transform=Compose([Graclus(2), PolarAdj()]))
test_dataset = MNISTSuperpixel75(
    path, train=False, transform=Compose([Graclus(2), PolarAdj()]))

train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True)

for batch, ((input, adj, position), target) in enumerate(test_loader):
    print(batch)
    # print(len(target))
    # print(input.size())
    # print('aa', adj[0][1].size())
    # print(adj[0][1])
