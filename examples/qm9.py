import os
import sys

sys.path.insert(0, '.')
sys.path.insert(0, '..')

from torch_geometric.datasets import QM9  # noqa

path = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(path, '..', 'data', 'QM9')

train_dataset = QM9(path, train=True)
test_dataset = QM9(path, train=False)
