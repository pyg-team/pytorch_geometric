import torch
from torch.autograd import Variable
from torch_scatter import scatter_max, scatter_mean, scatter_add

from .coalesce import remove_self_loops, coalesce


def _pool(index, position, cluster, weight):
    # Map edge indices to new cluster indices.
    index = index.contiguous()
    index = cluster[index.view(-1)].view(2, -1)
    index = remove_self_loops(index)  # Remove self loops.
    #if index.dim() == 1:
    #    index = index.view(2,1)
    index = coalesce(index)  # Remove duplicates.

    if weight is not None:
        print(weight.data.min(), weight.data.max())
        weight = torch.sigmoid(weight).unsqueeze(1)
        norm = scatter_add(Variable(cluster), weight.squeeze(), dim=0)
        norm = torch.gather(norm,0,Variable(cluster))

        weight = weight/norm.unsqueeze(1)
        cluster = cluster.unsqueeze(1).expand(-1, position.size(1))

        position *= weight
        position = scatter_add(Variable(cluster), position, dim=0)
    else:
        cluster = cluster.unsqueeze(1).expand(-1, position.size(1))
        position = scatter_mean(Variable(cluster), position, dim=0)

    return index, position


def _max_pool(input, cluster, size):
    if input.dim() == 1:
        input = input.unsqueeze(1)

    cluster = cluster.unsqueeze(1).expand(-1, input.size(1))
    cluster = cluster if torch.is_tensor(input) else Variable(cluster)

    fill = -999999
    if size is None:
        x = scatter_max(cluster, input, dim=0, fill_value=fill)[0]
    else:
        x = scatter_max(cluster, input, dim=0, size=size, fill_value=fill)[0]
        x[(x == fill).data] = 0
    return x


def max_pool(input, index, position, cluster, size=None, weight=None):
    x = _max_pool(input, cluster, size)
    return (x, ) + _pool(index, position, cluster, weight)
