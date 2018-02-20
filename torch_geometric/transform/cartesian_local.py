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
        print(weight.min(), weight.max())
        row_idx = row.unsqueeze(1).expand(row.size(0),weight.size(1))
        print(row_idx.size(),col.size(),weight.size())
        maximums_n = scatter_max(row_idx, weight, dim=0)[0].max(dim=1)[0]
        maximums_e = maximums_n.gather(dim=0, index=row)
        print(maximums_n.size(),maximums_e.size())
        print(maximums_e.min(),maximums_e.max())
        weight *= 1 / (2 * maximums_e.unsqueeze(1))
        weight += 0.5
        print(weight.min(), weight.max())

        if data.weight is None:
            data.weight = weight
        else:
            data.weight = torch.cat([weight, data.weight.unsqueeze(1)], dim=1)

        return data
