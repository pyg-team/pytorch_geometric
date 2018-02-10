import os
import sys

sys.path.insert(0, '.')
sys.path.insert(0, '..')

from torch_geometric.datasets import ShapeNet  # noqa
from torch_geometric.utils import DataLoader  # noqa
from torch_geometric.transforms import CartesianAdj  # noqa

path = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(path, '..', 'data', 'ShapeNet')

category = 'Airplane'
transform = CartesianAdj()
train_dataset = ShapeNet(path, category, True, transform)
test_dataset = ShapeNet(path, category, False, transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

for data in train_loader:
    data = data.cuda().to_variable()
    print(data.num_nodes)
