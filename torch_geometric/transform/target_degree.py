from __future__ import division

import torch

from ..sparse import SparseTensor, sum


class TargetDegreeAdj(object):
    def __call__(self, data):
        adj = data.adj
        index = adj._indices()
        col = index[1]
        n = adj.size(0)

        # Compute target input degree.
        degree = sum(adj, dim=0)  # Input degree.
        degree /= degree.max()
        degree = degree[col]

        # Modify data and return.
        data.adj = SparseTensor(index, degree, torch.Size([n, n]))
        return data
