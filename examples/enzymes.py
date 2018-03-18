from __future__ import division, print_function

import os.path as osp
import sys

sys.path.insert(0, '.')
sys.path.insert(0, '..')

from torch_geometric.datasets import ENZYMES  # noqa
from torch_geometric.utils import DataLoader2  # noqa

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'ENZYMES')
dataset = ENZYMES(path)
loader = DataLoader2(dataset, batch_size=32, shuffle=True)

for data in loader:
    data = data.cuda().to_variable()
    print(data.num_nodes, data.num_edges)
