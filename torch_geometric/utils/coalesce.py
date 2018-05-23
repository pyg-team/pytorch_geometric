import torch
from torch_unique import unique

from .num_nodes import maybe_num_nodes


def is_coalesced(edge_index, num_nodes=None):
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    row, col = edge_index

    index = num_nodes * row + col
    return index.size(0) == unique(index)[0].size(0)


def coalesce(edge_index, edge_attr=None, num_nodes=None):
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    row, col = edge_index

    if edge_attr is None:
        index = num_nodes * row + col
        _, perm = unique(index)
        edge_index = edge_index[:, perm]
    else:
        t = torch.cuda if edge_attr.is_cuda else torch
        sparse = getattr(t.sparse, edge_attr.type().split('.')[-1])
        n = num_nodes
        size = torch.Size([n, n] + list(edge_attr.size())[1:])
        adj = sparse(edge_index, edge_attr, size).coalesce()
        edge_index, edge_attr = adj._indices(), adj._values()

    return edge_index, edge_attr
