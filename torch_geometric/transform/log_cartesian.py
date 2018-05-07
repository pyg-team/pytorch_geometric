from __future__ import division

from math import log

import torch


class LogCartesian(object):
    def __init__(self, scale=1):
        self.scale = scale
        self.norm = 1 / log(1 + scale)

    def __call__(self, data):
        row, col = data.index

        # Compute Log-Cartesian pseudo-coordinates.
        weight = data.pos[col] - data.pos[row]
        mask = 1 - 2 * (weight < 0).type_as(weight)
        weight /= weight.abs().max()
        weight = weight.abs()
        weight = torch.log(self.scale * weight.abs() + 1) * self.norm
        weight *= mask
        weight *= 0.5
        weight += 0.5

        if data.weight is None:
            data.weight = weight
        else:
            data.weight = torch.cat([weight, data.weight.unsqueeze(1)], dim=1)

        return data
