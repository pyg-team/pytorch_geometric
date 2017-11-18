from __future__ import division

import torch

from .base import BaseAdj
from ..sparse import SparseTensor


class TargetDegreeAdj(BaseAdj):
    def _call(self, adj, position):
        n = adj.size(0)
        index = adj._indices()
        row, col = index
        weight = adj._values().float()

        degree = weight.new(n).fill_(0)
        degree.scatter_add_(0, col, weight)  # Collect input degree.
        degree /= degree.max()  # Normalize globally.
        target_degree = degree[col]

        adj = SparseTensor(index, target_degree, torch.Size([n, n]))

        return adj, position
