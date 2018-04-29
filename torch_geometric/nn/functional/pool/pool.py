import torch
from torch.autograd import Variable
from torch_scatter import scatter_mean, scatter_max

from .coalesce import remove_self_loops, coalesce


def pool(edge_index, cluster, edge_attr=None, pos=None):
    edge_index = cluster[edge_index.view(-1)].view(2, -1)
    edge_index = remove_self_loops(edge_index)
    edge_index = coalesce(edge_index)

    if edge_attr is not None:
        edge_attr = None  # TODO

    if pos is not None:
        pos = pos.unsqueeze(-1) if pos.dim() == 1 else pos
        _cluster = cluster.unsqueeze(-1).expand_as(pos)
        _cluster = _cluster if torch.is_tensor(pos) else Variable(cluster)
        pos = scatter_mean(_cluster, pos, dim=0)

    return edge_index, edge_attr, pos


def max_pool(x, cluster, size=None):
    fill = -9999999
    x, _ = scatter_max(x, cluster, dim=0, dim_size=size, fill_value=fill)

    if size is not None:
        x[(x == fill)] = 0

    return x
