from math import pi as PI

import torch

from ..sparse import SparseTensor


class PolarAdj(object):
    def __call__(self, data):
        input, adj, position = data
        index = adj._indices()
        n, dim = position.size()
        row, col = index

        direction = (position[col] - position[row]).float()

        rho = (direction * direction).sum(1).sqrt()
        rho /= rho.max()

        theta = torch.atan2(direction[:, 1], direction[:, 0]) / (2 * PI)
        theta += (theta < 0).type_as(theta)

        if dim == 3:
            phi = torch.acos(direction[:, 2]) / PI
            polar = torch.stack([rho, theta, phi], dim=1)
        else:
            polar = torch.stack([rho, theta], dim=1)

        adj = SparseTensor(index, polar, torch.Size([n, n, dim]))

        return input, adj, position
