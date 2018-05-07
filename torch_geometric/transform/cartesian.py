from __future__ import division
from torch.nn import Module, Parameter
import torch


class Cartesian(Module):
    """Concatenates Cartesian spatial relations based on the position
    :math:`P \in \mathbb{R}^{N x D}` of graph nodes to the graph's edge
    attributes."""

    def __init__(self, r=None, trainable=False):
        super(Cartesian, self).__init__()
        if r is not None:
            r = torch.FloatTensor([r]).cuda()
        if trainable and r is not None:
            self.r = Parameter(r)
        else:
            self.r = r

    def __call__(self, data):
        row, col = data.index
        # Compute Cartesian pseudo-coordinates.
        weight = data.pos[col] - data.pos[row]

        max = weight.abs().max() if self.r is None else self.r.clamp(
            min=0.0001)

        if self.r is not None:
            weight = weight * (1 / max)
            factor = weight.abs().max(1)[0].clamp(min=1)
            weight = weight / factor.unsqueeze(1)
            weight = weight / 2
        else:
            weight = weight * (1 / (2 * max))

        weight = weight + 0.5

        if data.weight is None:
            data.weight = weight
        else:
            data.weight = torch.cat([weight, data.weight.unsqueeze(1)], dim=1)

        return data
