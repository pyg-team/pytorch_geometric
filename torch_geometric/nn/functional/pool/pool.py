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


def _max_pool(input, cluster, size):
    if input.dim() == 1:
        input = input.unsqueeze(1)

    cluster = Variable(cluster.unsqueeze(1).expand(-1, input.size(1)))
    fill = -999999
    if size is None:
        x = scatter_max(cluster, input, dim=0, fill_value=fill)[0]
    else:
        x = scatter_max(cluster, input, dim=0, size=size, fill_value=fill)[0]
        mask = x == fill
        x[mask.data] = 0
    return x


def max_pool(input, index, position, cluster, size=None):
    x = _max_pool(input, cluster, size)
    return (x, ) + _pool(index, position, cluster)
