import sys

sys.path.insert(0, '.')
sys.path.insert(0, '..')

from torch_geometric.datasets import Cora  # noqa: E402

path = '~/Downloads/cora'
train_dataset = Cora(path)
