from __future__ import division

import torch


class CartesianAdj(object):
    """Concatenates Cartesian spatial relations based on the position
    :math:`P \in \mathbb{R}^{N x D}` of graph nodes to the graph's edge
    attributes."""

    def __init__(self, r=None):
        self.r = r

    def __call__(self, data):
        row, col = data.index

        # Compute Cartesian pseudo-coordinates.
        weight = data.pos[col] - data.pos[row]
        max = weight.abs().max() if self.r is None else self.r
        weight *= 1 / (2 * max)
        weight += 0.5

        if data.weight is None:
            data.weight = weight
        else:
            data.weight = torch.cat([weight, data.weight.unsqueeze(1)], dim=1)

        return data
