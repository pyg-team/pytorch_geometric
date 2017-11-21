from __future__ import division

import torch

from .base import BaseAdj
from ..sparse import SparseTensor, sum


class TargetIndegreeAdj(BaseAdj):
    """Converts a set of adjacency matrices :math:`A \in \mathbb{R}^{N x N}`
    saved in a dataset :obj:`Data` object to the globally normalized indegree
    of the target nodes
    :math:`u(i, j) = \deg(j)/\max_{i \in \mathcal{V}}(\deg(i))`.
    """

    def _call(self, adj, position):
        index = adj._indices()
        _, col = index
        n = adj.size(0)

        # Compute target input degree.
        degree = sum(adj, dim=0)  # Input degree.
        degree /= degree.max()  # Normalize.
        degree = degree[col]  # Target nodes.

        return SparseTensor(index, degree, torch.Size([n, n]))
