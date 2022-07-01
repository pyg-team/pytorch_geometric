from typing import List

import torch
from torch import Tensor

from torch_geometric.utils import degree


def unbatch_edge_index(edge_index: Tensor, batch: Tensor) -> List[Tensor]:
    r"""Splits the :obj:`edge_index` according to a :obj:`batch` vector.

    Args:
        edge_index (Tensor): The edge_index tensor.
        batch (LongTensor): The batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example. Must be ordered, consecutive, and
            starts with 0.

    :rtype: :class:`List[Tensor]`
    """
    boundary = torch.cumsum(degree(batch), dim=0)
    inc = torch.cat([boundary.new_tensor([0]), boundary[:-1]], dim=0)

    edge_assignments = torch.bucketize(edge_index, boundary, right=True)

    out = [(edge_index[edge_assignments == batch_idx].view(2, -1) -
            inc[batch_idx]).to(torch.int64)
           for batch_idx in range(batch.max().item() + 1)]

    return out
