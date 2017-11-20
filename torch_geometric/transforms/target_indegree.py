from __future__ import division

import torch

from .base import BaseAdj
from ..sparse import SparseTensor, sum


class TargetIndegreeAdj(BaseAdj):
    def _call(self, adj, position):
        index = adj._indices()
        _, col = index
        n = adj.size(0)

        # Compute target input degree.
        degree = sum(adj, dim=0)  # Input degree.
        degree /= degree.max()  # Normalize.
        degree = degree[col]  # Target nodes.

        return SparseTensor(index, degree, torch.Size([n, n]))
