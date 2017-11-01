from __future__ import division, print_function

import sys

sys.path.insert(0, '.')
sys.path.insert(0, '..')

from torch_geometric.datasets import MNISTSuperpixel75  # noqa
from torch_geometric.transform import PolarAdj  # noqa

path = '~/MNISTSuperpixel75'
train_dataset = MNISTSuperpixel75(path, train=True)
test_dataset = MNISTSuperpixel75(path, train=False)

print(len(train_dataset))

for i in range(70000):
    (input, position, adj), target = train_dataset[i]
    print(input.size())
    print(position.size())
    print(adj.size())
    print(target)
