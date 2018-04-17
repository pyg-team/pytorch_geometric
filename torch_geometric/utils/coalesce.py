import torch
from torch_unique import unique_by_key


def coalesce(edge_index, num_nodes=None):
    num_nodes = edge_index.max() + 1 if num_nodes is None else num_nodes
    row, col = edge_index
    idx = num_nodes * row + col

    perm = torch.arange(row.size(0), out=row.new())
    _, perm = unique_by_key(idx, perm)
    edge_index = edge_index[:, perm]

    return edge_index
