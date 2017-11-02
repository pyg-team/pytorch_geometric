from __future__ import division, print_function

import sys
from torchvision.transforms import Compose

sys.path.insert(0, '.')
sys.path.insert(0, '..')

from torch_geometric.datasets import MNISTSuperpixel75  # noqa
from torch_geometric.transform import Graclus, PolarAdj  # noqa

path = '~/MNISTSuperpixel75'
train_dataset = MNISTSuperpixel75(
    path, train=True, transform=Compose([Graclus(2), PolarAdj()]))
test_dataset = MNISTSuperpixel75(
    path, train=False, transform=Compose([Graclus(2), PolarAdj()]))

data, target = train_dataset[0]
print(len(data))
input, adj, position = data
print(input.size())
print(len(adj))
print(len(position))
print(adj[0].size())
