from __future__ import division

import torch
from torch_scatter import scatter_max


class CartesianLocalAdj(object):
    """Concatenates Cartesian spatial relations based on the position
    :math:`P \in \mathbb{R}^{N x D}` of graph nodes to the graph's edge
    attributes."""

    def __call__(self, data):
        (row, col), pos = data.index, data.pos
        num_nodes = pos.size(0)

        # Compute Cartesian pseudo-coordinates.
        weight = pos[col] - pos[row]

        # row_idx = row.unsqueeze(1).expand(row.size(0), weight.size(1))

        max_n, _ = scatter_max(weight.abs(), row, dim=0, dim_size=num_nodes)
        max_n, _ = max_n.max(dim=1)

        max_e = max_n.gather(dim=0, index=row)
        weight *= 1 / (2 * max_e.unsqueeze(1))
        weight += 0.5

        if data.weight is None:
            data.weight = weight
        else:
            data.weight = torch.cat([weight, data.weight.unsqueeze(1)], dim=1)

        return data
