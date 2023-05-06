from typing import Callable, Optional, Tuple

from torch import Tensor

from torch_geometric.data import Batch, Data
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from torch_geometric.nn.pool.pool import pool_batch, pool_edge, pool_pos
from torch_geometric.utils import add_self_loops, scatter


def _avg_pool_x(
    cluster: Tensor,
    x: Tensor,
    size: Optional[int] = None,
) -> Tensor:
    return scatter(x, cluster, dim=0, dim_size=size, reduce='mean')


def avg_pool_x(
    cluster: Tensor,
    x: Tensor,
    batch: Tensor,
    batch_size: Optional[int] = None,
    size: Optional[int] = None,
) -> Tuple[Tensor, Optional[Tensor]]:
    r"""Average pools node features according to the clustering defined in
    :attr:`cluster`.
    See :meth:`torch_geometric.nn.pool.max_pool_x` for more details.

    Args:
        cluster (torch.Tensor): The cluster vector
            :math:`\mathbf{c} \in \{ 0, \ldots, N - 1 \}^N`, which assigns each
            node to a specific cluster.
        x (Tensor): The node feature matrix.
        batch (torch.Tensor): The batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example.
        batch_size (int, optional): The number of examples :math:`B`.
            Automatically calculated if not given. (default: :obj:`None`)
        size (int, optional): The maximum number of clusters in a single
            example. (default: :obj:`None`)

    :rtype: (:class:`torch.Tensor`, :class:`torch.Tensor`) if :attr:`size` is
        :obj:`None`, else :class:`torch.Tensor`
    """
    if size is not None:
        if batch_size is None:
            batch_size = int(batch.max().item()) + 1
        return _avg_pool_x(cluster, x, batch_size * size), None

    cluster, perm = consecutive_cluster(cluster)
    x = _avg_pool_x(cluster, x)
    batch = pool_batch(perm, batch)

    return x, batch


def avg_pool(
    cluster: Tensor,
    data: Data,
    transform: Optional[Callable] = None,
) -> Data:
    r"""Pools and coarsens a graph given by the
    :class:`torch_geometric.data.Data` object according to the clustering
    defined in :attr:`cluster`.
    Final node features are defined by the *average* features of all nodes
    within the same cluster.
    See :meth:`torch_geometric.nn.pool.max_pool` for more details.

    Args:
        cluster (torch.Tensor): The cluster vector
            :math:`\mathbf{c} \in \{ 0, \ldots, N - 1 \}^N`, which assigns each
            node to a specific cluster.
        data (Data): Graph data object.
        transform (callable, optional): A function/transform that takes in the
            coarsened and pooled :obj:`torch_geometric.data.Data` object and
            returns a transformed version. (default: :obj:`None`)

    :rtype: :class:`torch_geometric.data.Data`
    """
    cluster, perm = consecutive_cluster(cluster)

    x = None if data.x is None else _avg_pool_x(cluster, data.x)
    index, attr = pool_edge(cluster, data.edge_index, data.edge_attr)
    batch = None if data.batch is None else pool_batch(perm, data.batch)
    pos = None if data.pos is None else pool_pos(cluster, data.pos)

    data = Batch(batch=batch, x=x, edge_index=index, edge_attr=attr, pos=pos)

    if transform is not None:
        data = transform(data)

    return data


def avg_pool_neighbor_x(
    data: Data,
    flow: Optional[str] = 'source_to_target',
) -> Data:
    r"""Average pools neighboring node features, where each feature in
    :obj:`data.x` is replaced by the average feature values from the central
    node and its neighbors.
    """
    x, edge_index = data.x, data.edge_index

    edge_index, _ = add_self_loops(edge_index, num_nodes=data.num_nodes)

    row, col = edge_index
    row, col = (row, col) if flow == 'source_to_target' else (col, row)

    data.x = scatter(x[row], col, dim=0, dim_size=data.num_nodes,
                     reduce='mean')
    return data
