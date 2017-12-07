from __future__ import division, print_function

import os
import sys

sys.path.insert(0, '.')
sys.path.insert(0, '..')

from torch_geometric.datasets import Cuneiform  # noqa
from torch_geometric.transforms import CartesianAdj  # noqa

path = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(path, '..', 'data', 'Cuneiform')
transform = CartesianAdj()
train_dataset = Cuneiform(path, train=True, transform=transform)
# test_dataset = Cuneiform(path, train=False, transform=transform)
