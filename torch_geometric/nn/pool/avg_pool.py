from torch_scatter import scatter_mean
from torch_geometric.data import Batch

from .consecutive import consecutive_cluster
from .pool import pool_edge, pool_batch, pool_pos


def _avg_pool_x(cluster, x, size=None):
    fill = -9999999
    x, _ = scatter_mean(x, cluster, dim=0, dim_size=size, fill_value=fill)
    x[x == fill] = 0
    return x


def avg_pool_x(cluster, x, batch=None, size=None):
    assert batch is None or size is None

    if size is not None:
        return _avg_pool_x(cluster, x, size)

    cluster, perm = consecutive_cluster(cluster)
    x = _avg_pool_x(cluster, x)
    batch = pool_batch(perm, batch)

    return x, batch


def avg_pool(cluster, data, transform=None):
    cluster, perm = consecutive_cluster(cluster)

    x = _avg_pool_x(cluster, data.x)
    index, attr = pool_edge(cluster, data.edge_index, data.edge_attr)
    batch = None if data.batch is None else pool_batch(perm, data.batch)
    pos = None if data.pos is None else pool_pos(cluster, data.pos)

    data = Batch(batch=batch, x=x, edge_index=index, edge_attr=attr, pos=pos)

    if transform is not None:
        data = transform(data)

    return data
