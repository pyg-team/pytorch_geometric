from torch_scatter import scatter_max, scatter_mean

from .coalesce import coalesce

def max_pool(input, index, position, cluster):
    input = scatter_max(cluster, input)
    position = scatter_mean(cluster, position)

    index = cluster[index.view(-1)].view(2, -1)  # Replace edge indices with cluster indices.
    mask = index[0, :] != index[1, :]  # Remove self-loops.
    index = coalesce(index)  # Remove duplicates.

    return input, index, position
