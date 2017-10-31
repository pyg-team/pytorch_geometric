import sys

sys.path.insert(0, '.')
sys.path.insert(0, '..')

from torch_geometric.datasets import CiteSeer  # noqa
from torch_geometric.transform import DegreeAdj  # noqa

CiteSeer('~/CiteSeer', transform=DegreeAdj())
