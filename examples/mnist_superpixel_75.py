from __future__ import division, print_function

import sys

sys.path.insert(0, '.')
sys.path.insert(0, '..')

from torch_geometric.datasets import MNISTSuperpixel75  # noqa
from torch_geometric.transform import PolarAdj  # noqa

path = '~/MNISTSuperpixel75'
train_dataset = MNISTSuperpixel75(path, train=True, transform=PolarAdj())
