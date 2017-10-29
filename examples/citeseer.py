import sys

sys.path.insert(0, '.')
sys.path.insert(0, '..')

from torch_geometric.datasets import CiteSeer  # noqa: E402

path = '~/Downloads/citeseer'
train_dataset = CiteSeer(path)
