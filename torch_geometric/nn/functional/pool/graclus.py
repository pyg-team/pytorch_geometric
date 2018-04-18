from torch_cluster import graclus_cluster

from .consecutive import consecutive_cluster
from .pool import pool, max_pool
from ....datasets.dataset import Data


def graclus_pool(data, weight=None, transform=None):
    row, col = data.index

    cluster = graclus_cluster(row, col, weight, data.num_nodes)
    cluster, batch = consecutive_cluster(cluster, data.batch)

    x = max_pool(data.input, cluster)
    edge_index, edge_attr, pos = pool(data.index, cluster, data.weight,
                                      data.pos)

    data = Data(x, pos, edge_index, edge_attr, data.target, batch)

    if transform is not None:
        data = transform(data)

    return data
