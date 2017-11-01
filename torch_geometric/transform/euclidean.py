from __future__ import division

import torch

from ..sparse import SparseTensor


class EuclideanAdj(object):
    def __call__(self, data):
        input, adj, position = data
        index = adj._indices()
        row, col = index

        direction = (position[col] - position[row]).float()
        c = 1 / (2 * direction.abs().max())
        direction = c * direction + 0.5
        n, dim = position.size()
        adj = SparseTensor(index, direction, torch.Size([n, n, dim]))

        return input, adj, position
