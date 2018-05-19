import torch
from torch_unique import unique_by_key

from .num_nodes import maybe_num_nodes


def coalesce(edge_index, num_nodes=None):
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    row, col = edge_index

    index = num_nodes * row + col
    perm = torch.arange(index.size(0), dtype=torch.long, device=row.device)
    _, perm = unique_by_key(index, perm)
    edge_index = edge_index[:, perm]

    return edge_index, perm
