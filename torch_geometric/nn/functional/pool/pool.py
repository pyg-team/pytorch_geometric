import torch
from torch_scatter import scatter_add

from .coalesce import remove_self_loops, coalesce


def _pool(index, position, cluster):
    index = cluster[index.view(-1)].view(2, -1)  # Replace edge indices with cluster indices.
    index = remove_self_loops(index)  # Remove self loops.
    index = coalesce(index)  # Remove duplicates.

    _cluster = cluster.unsqueeze(1).expand(-1, position.size(1))
    position = scatter_add(_cluster, position, dim=0)

    return index, position

def max_pool(input, index, position, cluster):
    _cluster = cluster.unsqueeze(1).expand(-1, input.size(1))
    input = scatter_add(_cluster, input, dim=0)
    return (input,) + _pool(index, position, cluster)


def average_pool(input, index, position, cluster):
    _cluster = cluster.unsqueeze(1).expand(-1, input.size(1))
    input = scatter_add(_cluster, input, dim=0)
    return (input,) + _pool(index, position, cluster)
