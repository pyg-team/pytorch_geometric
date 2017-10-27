from __future__ import division

import torch

from ..sparse import SparseTensor


class DegreeAdj(object):
    def __call__(self, data):
        input, adj = data
        index = adj._indices()
        row, col = index
        weight = adj._values().float()

        degree = weight.new(adj.size(0)).fill_(0)

        degree.scatter_add_(0, row, weight)
        degree = torch.sqrt(degree)
        degree = 1 / degree

        degree_row = degree[row]
        degree_col = degree[col]

        value = torch.stack([degree_row, degree_col], dim=1)

        n = adj.size(0)
        adj = SparseTensor(index, value, torch.Size([n, n, 2]))

        return input, adj
