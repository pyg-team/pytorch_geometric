from typing import Optional, Tuple

import torch
from torch import Tensor

from torch_geometric.nn.pool.select import SelectOutput
from torch_geometric.utils.num_nodes import maybe_num_nodes

from .base import Connect


class TopkConnect(Connect):
    def forward(
            cluster: SelectOutput, edge_index: Tensor,
            edge_attr: Optional[Tensor]) -> Tuple[Tensor, Optional[Tensor]]:
        return filter_adj(
            edge_index,
            edge_attr,
            cluster.node_index,
            cluster.num_nodes,
        )


def filter_adj(
    edge_index: Tensor,
    edge_attr: Optional[Tensor],
    perm: Tensor,
    num_nodes: Optional[int] = None,
) -> Tuple[Tensor, Optional[Tensor]]:
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    mask = perm.new_full((num_nodes, ), -1)
    i = torch.arange(perm.size(0), dtype=torch.long, device=perm.device)
    mask[perm] = i

    row, col = edge_index[0], edge_index[1]
    row, col = mask[row], mask[col]
    mask = (row >= 0) & (col >= 0)
    row, col = row[mask], col[mask]

    if edge_attr is not None:
        edge_attr = edge_attr[mask]

    return torch.stack([row, col], dim=0), edge_attr
