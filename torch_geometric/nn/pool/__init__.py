r"""Pooling package."""

import warnings
from typing import Optional
from torch import Tensor

import torch_geometric.typing
from torch_geometric.typing import OptTensor, torch_cluster

from .asap import ASAPooling
from .avg_pool import avg_pool, avg_pool_neighbor_x, avg_pool_x
from .edge_pool import EdgePooling
from .glob import global_add_pool, global_max_pool, global_mean_pool
from .knn import (KNNIndex, L2KNNIndex, MIPSKNNIndex, ApproxL2KNNIndex,
                  ApproxMIPSKNNIndex)
from .graclus import graclus
from .max_pool import max_pool, max_pool_neighbor_x, max_pool_x
from .mem_pool import MemPooling
from .pan_pool import PANPooling
from .sag_pool import SAGPooling
from .topk_pool import TopKPooling
from .voxel_grid import voxel_grid
from .approx_knn import approx_knn, approx_knn_graph


def fps(
    x: Tensor,
    batch: OptTensor = None,
    ratio: float = 0.5,
    random_start: bool = True,
    batch_size: Optional[int] = None,
) -> Tensor:
    r"""A sampling algorithm from the `"PointNet++: Deep Hierarchical Feature
    Learning on Point Sets in a Metric Space"
    <https://arxiv.org/abs/1706.02413>`_ paper, which iteratively samples the
    most distant point with regard to the rest points.

    .. code-block:: python

        import torch
        from torch_geometric.nn import fps

        x = torch.tensor([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
        batch = torch.tensor([0, 0, 0, 0])
        index = fps(x, batch, ratio=0.5)

    Args:
        x (torch.Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
        batch (torch.Tensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example. (default: :obj:`None`)
        ratio (float, optional): Sampling ratio. (default: :obj:`0.5`)
        random_start (bool, optional): If set to :obj:`False`, use the first
            node in :math:`\mathbf{X}` as starting node. (default: obj:`True`)
        batch_size (int, optional): The number of examples :math:`B`.
            Automatically calculated if not given. (default: :obj:`None`)

    :rtype: :class:`torch.Tensor`
    """
    if not torch_geometric.typing.WITH_TORCH_CLUSTER_BATCH_SIZE:
        return torch_cluster.fps(x, batch, ratio, random_start)
    return torch_cluster.fps(x, batch, ratio, random_start, batch_size)


def knn(
    x: Tensor,
    y: Tensor,
    k: int,
    batch_x: OptTensor = None,
    batch_y: OptTensor = None,
    cosine: bool = False,
    num_workers: int = 1,
    batch_size: Optional[int] = None,
) -> Tensor:
    r"""Finds for each element in :obj:`y` the :obj:`k` nearest points in
    :obj:`x`.

    .. code-block:: python

        import torch
        from torch_geometric.nn import knn

        x = torch.tensor([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
        batch_x = torch.tensor([0, 0, 0, 0])
        y = torch.tensor([[-1.0, 0.0], [1.0, 0.0]])
        batch_y = torch.tensor([0, 0])
        assign_index = knn(x, y, 2, batch_x, batch_y)

    Args:
        x (torch.Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
        y (torch.Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{M \times F}`.
        k (int): The number of neighbors.
        batch_x (torch.Tensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example. (default: :obj:`None`)
        batch_y (torch.Tensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^M`, which assigns each
            node to a specific example. (default: :obj:`None`)
        cosine (bool, optional): If :obj:`True`, will use the cosine
            distance instead of euclidean distance to find nearest neighbors.
            (default: :obj:`False`)
        num_workers (int, optional): Number of workers to use for computation.
            Has no effect in case :obj:`batch_x` or :obj:`batch_y` is not
            :obj:`None`, or the input lies on the GPU. (default: :obj:`1`)
        batch_size (int, optional): The number of examples :math:`B`.
            Automatically calculated if not given. (default: :obj:`None`)

    :rtype: :class:`torch.Tensor`
    """
    if not torch_geometric.typing.WITH_TORCH_CLUSTER_BATCH_SIZE:
        return torch_cluster.knn(x, y, k, batch_x, batch_y, cosine,
                                 num_workers)
    return torch_cluster.knn(x, y, k, batch_x, batch_y, cosine, num_workers,
                             batch_size)


def knn_graph(
    x: Tensor,
    k: int,
    batch: OptTensor = None,
    loop: bool = False,
    flow: str = 'source_to_target',
    cosine: bool = False,
    num_workers: int = 1,
    batch_size: Optional[int] = None,
) -> Tensor:
    r"""Computes graph edges to the nearest :obj:`k` points.

    .. code-block:: python

        import torch
        from torch_geometric.nn import knn_graph

        x = torch.tensor([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
        batch = torch.tensor([0, 0, 0, 0])
        edge_index = knn_graph(x, k=2, batch=batch, loop=False)

    Args:
        x (torch.Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
        k (int): The number of neighbors.
        batch (torch.Tensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example. (default: :obj:`None`)
        loop (bool, optional): If :obj:`True`, the graph will contain
            self-loops. (default: :obj:`False`)
        flow (str, optional): The flow direction when using in combination with
            message passing (:obj:`"source_to_target"` or
            :obj:`"target_to_source"`). (default: :obj:`"source_to_target"`)
        cosine (bool, optional): If :obj:`True`, will use the cosine
            distance instead of euclidean distance to find nearest neighbors.
            (default: :obj:`False`)
        num_workers (int, optional): Number of workers to use for computation.
            Has no effect in case :obj:`batch` is not :obj:`None`, or the input
            lies on the GPU. (default: :obj:`1`)
        batch_size (int, optional): The number of examples :math:`B`.
            Automatically calculated if not given. (default: :obj:`None`)

    :rtype: :class:`torch.Tensor`
    """
    if batch is not None and x.device != batch.device:
        warnings.warn("Input tensor 'x' and 'batch' are on different devices "
                      "in 'knn_graph'. Performing blocking device transfer")
        batch = batch.to(x.device)

    if not torch_geometric.typing.WITH_TORCH_CLUSTER_BATCH_SIZE:
        return torch_cluster.knn_graph(x, k, batch, loop, flow, cosine,
                                       num_workers)
    return torch_cluster.knn_graph(x, k, batch, loop, flow, cosine,
                                   num_workers, batch_size)


def radius(
    x: Tensor,
    y: Tensor,
    r: float,
    batch_x: OptTensor = None,
    batch_y: OptTensor = None,
    max_num_neighbors: int = 32,
    num_workers: int = 1,
    batch_size: Optional[int] = None,
) -> Tensor:
    r"""Finds for each element in :obj:`y` all points in :obj:`x` within
    distance :obj:`r`.

    .. code-block:: python

        import torch
        from torch_geometric.nn import radius

        x = torch.tensor([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
        batch_x = torch.tensor([0, 0, 0, 0])
        y = torch.tensor([[-1.0, 0.0], [1.0, 0.0]])
        batch_y = torch.tensor([0, 0])
        assign_index = radius(x, y, 1.5, batch_x, batch_y)

    Args:
        x (torch.Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
        y (torch.Tensor): Node feature matrix
            :math:`\mathbf{Y} \in \mathbb{R}^{M \times F}`.
        r (float): The radius.
        batch_x (torch.Tensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example. (default: :obj:`None`)
        batch_y (torch.Tensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^M`, which assigns each
            node to a specific example. (default: :obj:`None`)
        max_num_neighbors (int, optional): The maximum number of neighbors to
            return for each element in :obj:`y`. (default: :obj:`32`)
        num_workers (int, optional): Number of workers to use for computation.
            Has no effect in case :obj:`batch_x` or :obj:`batch_y` is not
            :obj:`None`, or the input lies on the GPU. (default: :obj:`1`)
        batch_size (int, optional): The number of examples :math:`B`.
            Automatically calculated if not given. (default: :obj:`None`)

    :rtype: :class:`torch.Tensor`

    .. warning::

        The CPU implementation of :meth:`radius` with :obj:`max_num_neighbors`
        is biased towards certain quadrants.
        Consider setting :obj:`max_num_neighbors` to :obj:`None` or moving
        inputs to GPU before proceeding.
    """
    if not torch_geometric.typing.WITH_TORCH_CLUSTER_BATCH_SIZE:
        return torch_cluster.radius(x, y, r, batch_x, batch_y,
                                    max_num_neighbors, num_workers)
    return torch_cluster.radius(x, y, r, batch_x, batch_y, max_num_neighbors,
                                num_workers, batch_size)


def radius_graph(
    x: Tensor,
    r: float,
    batch: OptTensor = None,
    loop: bool = False,
    max_num_neighbors: int = 32,
    flow: str = 'source_to_target',
    num_workers: int = 1,
    batch_size: Optional[int] = None,
) -> Tensor:
    r"""Computes graph edges to all points within a given distance.

    .. code-block:: python

        import torch
        from torch_geometric.nn import radius_graph

        x = torch.tensor([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
        batch = torch.tensor([0, 0, 0, 0])
        edge_index = radius_graph(x, r=1.5, batch=batch, loop=False)

    Args:
        x (torch.Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
        r (float): The radius.
        batch (torch.Tensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example. (default: :obj:`None`)
        loop (bool, optional): If :obj:`True`, the graph will contain
            self-loops. (default: :obj:`False`)
        max_num_neighbors (int, optional): The maximum number of neighbors to
            return for each element in :obj:`y`. (default: :obj:`32`)
        flow (str, optional): The flow direction when using in combination with
            message passing (:obj:`"source_to_target"` or
            :obj:`"target_to_source"`). (default: :obj:`"source_to_target"`)
        num_workers (int, optional): Number of workers to use for computation.
            Has no effect in case :obj:`batch` is not :obj:`None`, or the input
            lies on the GPU. (default: :obj:`1`)
        batch_size (int, optional): The number of examples :math:`B`.
            Automatically calculated if not given. (default: :obj:`None`)

    :rtype: :class:`torch.Tensor`

    .. warning::

        The CPU implementation of :meth:`radius_graph` with
        :obj:`max_num_neighbors` is biased towards certain quadrants.
        Consider setting :obj:`max_num_neighbors` to :obj:`None` or moving
        inputs to GPU before proceeding.
    """
    if batch is not None and x.device != batch.device:
        warnings.warn("Input tensor 'x' and 'batch' are on different devices "
                      "in 'radius_graph'. Performing blocking device transfer")
        batch = batch.to(x.device)

    if not torch_geometric.typing.WITH_TORCH_CLUSTER_BATCH_SIZE:
        return torch_cluster.radius_graph(x, r, batch, loop, max_num_neighbors,
                                          flow, num_workers)
    return torch_cluster.radius_graph(x, r, batch, loop, max_num_neighbors,
                                      flow, num_workers, batch_size)


def nearest(
    x: Tensor,
    y: Tensor,
    batch_x: OptTensor = None,
    batch_y: OptTensor = None,
) -> Tensor:
    r"""Finds for each element in :obj:`y` the :obj:`k` nearest point in
    :obj:`x`.

    .. code-block:: python

        import torch
        from torch_geometric.nn import nearest

        x = torch.tensor([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
        batch_x = torch.tensor([0, 0, 0, 0])
        y = torch.tensor([[-1.0, 0.0], [1.0, 0.0]])
        batch_y = torch.tensor([0, 0])
        cluster = nearest(x, y, batch_x, batch_y)

    Args:
        x (torch.Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
        y (torch.Tensor): Node feature matrix
            :math:`\mathbf{Y} \in \mathbb{R}^{M \times F}`.
        batch_x (torch.Tensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example. (default: :obj:`None`)
        batch_y (torch.Tensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^M`, which assigns each
            node to a specific example. (default: :obj:`None`)

    :rtype: :class:`torch.Tensor`
    """
    return torch_cluster.nearest(x, y, batch_x, batch_y)


__all__ = [
    'global_add_pool',
    'global_mean_pool',
    'global_max_pool',
    'KNNIndex',
    'L2KNNIndex',
    'MIPSKNNIndex',
    'ApproxL2KNNIndex',
    'ApproxMIPSKNNIndex',
    'TopKPooling',
    'SAGPooling',
    'EdgePooling',
    'ASAPooling',
    'PANPooling',
    'MemPooling',
    'max_pool',
    'avg_pool',
    'max_pool_x',
    'max_pool_neighbor_x',
    'avg_pool_x',
    'avg_pool_neighbor_x',
    'graclus',
    'voxel_grid',
    'fps',
    'knn',
    'knn_graph',
    'approx_knn',
    'approx_knn_graph',
    'radius',
    'radius_graph',
    'nearest',
]

classes = __all__
