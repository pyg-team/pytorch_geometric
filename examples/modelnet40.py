import os
import sys

sys.path.insert(0, '.')
sys.path.insert(0, '..')

from torch_geometric.datasets import ModelNet40  # noqa
from torch_geometric.utils import DataLoader2  # noqa
from torch_geometric.transform import CartesianAdj  # noqa

path = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(path, '..', 'data', 'ModelNet40')

transform = CartesianAdj()
train_dataset = ModelNet40(path, True, transform=transform)
test_dataset = ModelNet40(path, False, transform=transform)
train_loader = DataLoader2(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader2(test_dataset, batch_size=16)

for data in train_loader:
    data = data.cuda().to_variable()
    print(data.num_nodes, data.num_edges, data.adj.size())
