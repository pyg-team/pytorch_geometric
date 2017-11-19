from __future__ import division

import torch

from .base import BaseTransform
from ..sparse import SparseTensor, sum


class TargetIndegreeAdj(BaseTransform):
    def _call(self, data):
        adj = data.adj
        index = adj._indices()
        _, col = index
        n = adj.size(0)

        # Compute target input degree.
        degree = sum(adj, dim=0)  # Input degree.
        degree /= degree.max()  # Normalize.
        degree = degree[col]  # Target nodes.

        # Modify data and return.
        data.adj = SparseTensor(index, degree, torch.Size([n, n]))
        return data
