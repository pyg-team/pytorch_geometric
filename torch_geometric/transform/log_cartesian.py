from math import log

import torch


class LogCartesian(object):
    def __init__(self, scale=1):
        self.scale = scale
        self.norm = 1 / log(1 + scale)

    def __call__(self, data):
        row, col = data.edge_index

        # Compute Log-Cartesian pseudo-coordinates.
        weight = data.pos[col] - data.pos[row]
        mask = 1 - 2 * (weight < 0).type_as(weight)
        weight /= weight.abs().max()
        weight = weight.abs()
        weight = torch.log(self.scale * weight.abs() + 1) * self.norm
        weight *= mask
        weight *= 0.5
        weight += 0.5

        if data.edge_attr is None:
            data.edge_attr = weight
        else:
            data.edge_attr = torch.cat(
                [weight, data.weight.unsqueeze(1)], dim=1)

        return data
