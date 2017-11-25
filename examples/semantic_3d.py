from __future__ import division, print_function

import os
import sys

sys.path.insert(0, '.')
sys.path.insert(0, '..')

from torch_geometric.datasets import Semantic3D  # noqa

path = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(path, '..', 'data', 'Semantic3D')
train_dataset = Semantic3D(path, train=True)
# test_dataset = Semantic3D(path, train=False)
