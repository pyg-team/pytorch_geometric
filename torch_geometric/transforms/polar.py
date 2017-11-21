from math import pi as PI

import torch

from .base import BaseAdj
from ..sparse import SparseTensor


class PolarAdj(BaseAdj):
    """Converts a set of adjacency matrices :math:`A \in \mathbb{R}^{N x N}`
    saved in a dataset :obj:`Data` object to globally normalized polar
    pseudo-coordinates :math:`A \in [0, 1]^{N x N x 2}` based on the position
    :math:`P \in \mathbb{R}^{N x 2}` of graph-nodes.
    """

    def _call(self, adj, position):
        index = adj._indices()
        row, col = index
        n = adj.size(0)

        # Compute polar pseudo-coordinates.
        direction = position[col] - position[row]
        rho = (direction * direction).sum(1).sqrt()
        rho /= rho.max()
        theta = torch.atan2(direction[:, 1], direction[:, 0]) / (2 * PI)
        theta += (theta < 0).type_as(theta)
        polar = torch.stack([rho, theta], dim=1)

        return SparseTensor(index, polar, torch.Size([n, n, 2]))
