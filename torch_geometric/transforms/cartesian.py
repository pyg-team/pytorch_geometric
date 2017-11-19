from __future__ import division

import torch

from ..sparse import SparseTensor


class CartesianAdj(object):
    def __call__(self, data):
        adj, position = data.adj, data.position
        index = adj._indices()
        row, col = index
        n, dim = position.size()

        # Compute Cartesian pseudo-coordinates.
        cartesian = position[col] - position[row]
        cartesian *= 1 / (2 * cartesian.abs().max())
        cartesian += 0.5

        # Modify data and return.
        data.adj = SparseTensor(index, cartesian, torch.Size([n, n, dim]))
        return data
