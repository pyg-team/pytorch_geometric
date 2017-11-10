from __future__ import division, print_function

import sys

import torch

sys.path.insert(0, '.')
sys.path.insert(0, '..')

from torch_geometric.datasets import ShapeNet  # noqa

path = '~/ShapeNet'
train_dataset = ShapeNet(path, train=True)
