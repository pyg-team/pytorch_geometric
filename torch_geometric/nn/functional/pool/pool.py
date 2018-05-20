from torch_scatter import scatter_mean, scatter_max
from torch_geometric.utils import remove_self_loops, coalesce


def pool(edge_index, cluster, edge_attr=None, pos=None):
    edge_index = cluster[edge_index.view(-1)].view(2, -1)
    edge_index = remove_self_loops(edge_index)
    edge_index, _ = coalesce(edge_index, num_nodes=cluster.size(0))

    if edge_attr is not None:
        edge_attr = None  # TODO

    if pos is not None:
        pos = scatter_mean(pos, cluster, dim=0)

    return edge_index, edge_attr, pos


def max_pool(x, cluster, size=None):
    fill = -9999999
    x, _ = scatter_max(x, cluster, dim=0, dim_size=size, fill_value=fill)
    x[x == fill] = 0
    return x
