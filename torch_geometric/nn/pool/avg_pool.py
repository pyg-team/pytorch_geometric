from torch_geometric.data import Batch
from torch_geometric.utils import scatter_

from .consecutive import consecutive_cluster
from .pool import pool_edge, pool_batch, pool_pos


def _avg_pool_x(cluster, x, size=None):
    return scatter_('mean', x, cluster, dim_size=size)


def avg_pool_x(cluster, x, batch, size=None):
    if size is not None:
        return _avg_pool_x(cluster, x, (batch.max().item() + 1) * size)

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
