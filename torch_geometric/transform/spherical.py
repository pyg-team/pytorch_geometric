from __future__ import division

from math import pi as PI

import torch


class Spherical(object):
    def __call__(self, data):

        index = data.index
        row, col = index

        # Compute spherical pseudo-coordinates.
        direction = data.pos[col] - data.pos[row]
        rho = (direction * direction).sum(1).sqrt()
        rho /= rho.max()
        theta = torch.atan2(direction[:, 1], direction[:, 0]) / (2 * PI)
        theta += (theta < 0).type_as(theta)
        phi = torch.acos(direction[:, 2]) / PI
        spherical = torch.stack([rho, theta, phi], dim=1)

        if data.weight is None:
            data.weight = spherical
        else:
            data.weight = torch.cat(
                [spherical, data.weight.unsqueeze(1)], dim=1)

        return data
