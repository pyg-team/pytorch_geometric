from __future__ import division, print_function

import sys

sys.path.insert(0, '.')
sys.path.insert(0, '..')

from torch_geometric.datasets import MNISTSuperpixel75  # noqa
from torch_geometric.transform import PolarAdj  # noqa

path = '~/MNISTSuperpixel75'
train_dataset = MNISTSuperpixel75(path, train=True, transform=PolarAdj())
test_dataset = MNISTSuperpixel75(path, train=False, transform=PolarAdj())

print(len(train_dataset))
print(len(test_dataset))

(input, adj, position), target = train_dataset[0]
print(input.size())
print(adj.size())
print(position.size())
(input, adj, position), target = test_dataset[0]
print(input.size())
print(adj.size())
print(position.size())
