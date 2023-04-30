from typing import Optional, Tuple

import torch
from torch import Tensor

from torch_geometric.nn.pool.connect import Connect
from torch_geometric.utils.num_nodes import maybe_num_nodes


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


class TopKConnect(Connect):
    def __init__(self):
        super().__init__()

    def forward(cluster, edge_index, num_nodes, edge_attr, batch):
        perm = (cluster != -1).nonzero(as_tuple=False).view(-1)
        return filter_adj(edge_index, edge_attr, perm, num_nodes)
