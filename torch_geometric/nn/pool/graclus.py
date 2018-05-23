from torch_cluster import graclus_cluster
from torch_geometric.data import Batch

from .consecutive import consecutive_cluster
from .pool import pool, max_pool


def graclus_pool(data, edge_attr=None, transform=None):
    row, col = data.edge_index

    cluster = graclus_cluster(row, col, edge_attr, data.num_nodes)
    cluster, batch = consecutive_cluster(cluster, data.batch)

    x = max_pool(data.x, cluster)
    edge_index, _, pos = pool(data.edge_index, cluster, None, data.pos)

    data = Batch(x=x, edge_index=edge_index, pos=pos, batch=batch)

    if transform is not None:
        data = transform(data)

    return data
