import os
import sys

from torchvision.transforms import Compose

sys.path.insert(0, '.')
sys.path.insert(0, '..')

from torch_geometric.datasets import ModelNet10PC  # noqa
from torch_geometric.utils import DataLoader2  # noqa
from torch_geometric.transform import NormalizeScale, CartesianAdj  # noqa

path = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(path, '..', 'data', 'ModelNet10PC')

transform = Compose([NormalizeScale(), CartesianAdj()])
train_dataset = ModelNet10PC(path, True, transform=transform)
test_dataset = ModelNet10PC(path, False, transform=transform)
train_loader = DataLoader2(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader2(test_dataset, batch_size=32)

for data in train_loader:
    data = data.cuda().to_variable()
    print(data.num_nodes, data.num_edges, data.adj.size())
