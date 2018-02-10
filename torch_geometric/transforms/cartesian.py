from __future__ import division

import torch

from .base import BaseAdj
from ..sparse import SparseTensor


class CartesianAdj(BaseAdj):
    """Converts a set of adjacency matrices :math:`A \in \mathbb{R}^{N x N}`
    saved in a dataset :obj:`Data` object to globally normalized Cartesian
    pseudo-coordinates :math:`A \in [0, 1]^{N x N x d}` based on the position
    :math:`P \in \mathbb{R}^{N x d}` of graph-nodes.
    """

    def _call(self, adj, position):
        index = adj._indices()
        row, col = index
        n, dim = position.size()

        # Compute Cartesian pseudo-coordinates.
        cartesian = position[col] - position[row]
        cartesian *= 1 / (2 * cartesian.abs().max())
        cartesian += 0.5

        return SparseTensor(index, cartesian, torch.Size([n, n, dim]))
