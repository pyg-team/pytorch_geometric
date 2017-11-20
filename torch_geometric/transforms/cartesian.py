from __future__ import division

import torch

from .base import BaseAdj
from ..sparse import SparseTensor


class CartesianAdj(BaseAdj):
    def _call(self, adj, position):
        index = adj._indices()
        row, col = index
        n, dim = position.size()

        # Compute Cartesian pseudo-coordinates.
        cartesian = position[col] - position[row]
        cartesian *= 1 / (2 * cartesian.abs().max())
        cartesian += 0.5

        return SparseTensor(index, cartesian, torch.Size([n, n, dim]))
