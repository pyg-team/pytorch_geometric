from __future__ import division

import torch

from .base import BaseAdj
from ..sparse import SparseTensor


class EuclideanAdj(BaseAdj):
    def _call(self, adj, position):
        index = adj._indices()
        row, col = index

        direction = (position[col] - position[row]).float()
        c = 1 / (2 * direction.abs().max())
        direction = c * direction + 0.5
        n, dim = position.size()
        adj = SparseTensor(index, direction, torch.Size([n, n, dim]))

        return adj, position
