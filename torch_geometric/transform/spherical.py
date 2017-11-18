from math import pi as PI

import torch

from .base import BaseAdj
from ..sparse import SparseTensor


class SphericalAdj(BaseAdj):
    def _call(self, adj, position):
        index = adj._indices()
        n, dim = position.size()
        row, col = index

        direction = (position[col] - position[row]).float()

        rho = (direction * direction).sum(1).sqrt()
        rho /= rho.max()

        theta = torch.atan2(direction[:, 1], direction[:, 0]) / (2 * PI)
        theta += (theta < 0).type_as(theta)

        phi = torch.acos(direction[:, 2]) / PI

        spherical = torch.stack([rho, theta, phi], dim=1)
        adj = SparseTensor(index, spherical, torch.Size([n, n, dim]))

        return adj, position
