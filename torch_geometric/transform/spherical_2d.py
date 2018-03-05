from __future__ import division

from math import pi as PI

import torch


class Spherical2dAdj(object):
    """Concatenates Cartesian spatial relations based on the position
    :math:`P \in \mathbb{R}^{N x D}` of graph nodes to the graph's edge
    attributes."""

    def __call__(self, data):

        index = data.index
        row, col = index

        # Compute spherical pseudo-coordinates.
        direction = data.pos[col] - data.pos[row]
        theta = torch.atan2(direction[:, 1], direction[:, 0]) / (2 * PI)
        theta += (theta < 0).type_as(theta)
        phi = torch.acos(direction[:, 2]) / PI
        spherical = torch.stack([theta, phi], dim=1)

        if data.weight is None:
            data.weight = spherical
        else:
            data.weight = torch.cat([spherical, data.weight.unsqueeze(1)], dim=1)

        return data

