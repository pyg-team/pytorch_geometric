import torch
from torch.autograd import Variable
from torch_scatter import scatter_max, scatter_mean

from .coalesce import remove_self_loops, coalesce


def _pool(index, position, cluster):
    # Map edge indices to new cluster indices.
    index = cluster[index.view(-1)].view(2, -1)
    index = remove_self_loops(index)  # Remove self loops.
    index = coalesce(index)  # Remove duplicates.

    cluster = cluster.unsqueeze(1).expand(-1, position.size(1))
    position = scatter_mean(cluster, position, dim=0)

    return index, position


def _max_pool(input, cluster):
    fill = -999999
    cluster = cluster.unsqueeze(1).expand(-1, input.size(1))
    if torch.is_tensor(input):
        return scatter_max(cluster, input, dim=0, fill_value=fill)[0]
    else:
        return scatter_max(Variable(cluster), input, dim=0, fill_value=fill)[0]


def max_pool(input, index, position, cluster):
    return (_max_pool(input, cluster), ) + _pool(index, position, cluster)


def _avg_pool(input, cluster):
    cluster = cluster.unsqueeze(1).expand(-1, input.size(1))
    if torch.is_tensor(input):
        return scatter_mean(cluster, input, dim=0)
    else:
        return scatter_mean(Variable(cluster), input, dim=0)


def avg_pool(input, index, position, cluster):
    return (_avg_pool(input, cluster), ) + _pool(index, position, cluster)
