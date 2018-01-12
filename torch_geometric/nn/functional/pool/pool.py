import torch
from torch.autograd import Variable
from torch_scatter import scatter_add

from .coalesce import remove_self_loops, coalesce


def _pool(index, position, cluster):
    index = cluster[index.view(-1)].view(2, -1)  # Replace edge indices with cluster indices.
    index = remove_self_loops(index)  # Remove self loops.
    index = coalesce(index)  # Remove duplicates.

    cluster = cluster.unsqueeze(1).expand(-1, position.size(1))
    position = scatter_add(cluster, position, dim=0)

    return index, position


def _max_pool(input, cluster):
    cluster = cluster.unsqueeze(1).expand(-1, input.size(1))
    if torch.is_tensor(input):
        return scatter_add(cluster, input, dim=0)
    else:
        return scatter_add(Variable(cluster), input, dim=0)


def max_pool(input, index, position, cluster):
    return (_max_pool(input, cluster),) + _pool(index, position, cluster)


def _avg_pool(input, cluster):
    cluster = cluster.unsqueeze(1).expand(-1, input.size(1))
    if torch.is_tensor(input):
        return scatter_add(cluster, input, dim=0)
    else:
        return scatter_add(Variable(cluster), input, dim=0)


def avg_pool(input, index, position, cluster):
    return (_avg_pool(input, cluster),) + _pool(index, position, cluster)
