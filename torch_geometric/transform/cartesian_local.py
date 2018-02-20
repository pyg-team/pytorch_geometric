from __future__ import division

import torch
from torch_scatter import scatter_max


class CartesianLocalAdj(object):
    """Concatenates Cartesian spatial relations based on the position
    :math:`P \in \mathbb{R}^{N x D}` of graph nodes to the graph's edge
    attributes."""

    def __call__(self, data):
        row, col = data.index
        # Compute Cartesian pseudo-coordinates.
        weight = data.pos[col] - data.pos[row]

        row_idx = row.unsqueeze(1).expand(row.size(0), weight.size(1))

        max_n, _ = scatter_max(row_idx, weight, dim=0, fill_value=-9999999)
        max_n, _ = max_n.abs().max(dim=1)

        max_e = max_n.gather(dim=0, index=row)

        weight *= 1 / (2 * max_e.unsqueeze(1))
        weight += 0.5

        if data.weight is None:
            data.weight = weight
        else:
            data.weight = torch.cat([weight, data.weight.unsqueeze(1)], dim=1)

        return data
