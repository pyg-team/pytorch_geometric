import torch
from torch import Tensor


def approx_knn(
    x: Tensor,
    y: Tensor,
    k: int,
    batch_x: Tensor = None,
    batch_y: Tensor = None,
) -> Tensor:  # pragma: no cover
    r"""Finds for each element in :obj:`y` the :obj:`k` approximated nearest
    points in :obj:`x`.

    .. note::

        Approximated :math:`k`-nearest neighbor search is performed via the
        `pynndescent <https://pynndescent.readthedocs.io/en/latest>`_ library.

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

    :rtype: :class:`torch.Tensor`
    """
    from pynndescent import NNDescent

    if batch_x is None:
        batch_x = x.new_zeros(x.size(0), dtype=torch.long)
    if batch_y is None:
        batch_y = y.new_zeros(y.size(0), dtype=torch.long)

    x = x.view(-1, 1) if x.dim() == 1 else x
    y = y.view(-1, 1) if y.dim() == 1 else y

    assert x.dim() == 2 and batch_x.dim() == 1
    assert y.dim() == 2 and batch_y.dim() == 1
    assert x.size(1) == y.size(1)
    assert x.size(0) == batch_x.size(0)
    assert y.size(0) == batch_y.size(0)

    min_xy = min(x.min(), y.min())
    x, y = x - min_xy, y - min_xy

    max_xy = max(x.max(), y.max())
    x, y, = x / max_xy, y / max_xy

    # Concat batch/features to ensure no cross-links between examples exist:
    x = torch.cat([x, 2 * x.size(1) * batch_x.view(-1, 1).to(x.dtype)], dim=-1)
    y = torch.cat([y, 2 * y.size(1) * batch_y.view(-1, 1).to(y.dtype)], dim=-1)

    index = NNDescent(x.detach().cpu().numpy())
    col, dist = index.query(y.detach().cpu().numpy(), k=k)
    dist = torch.from_numpy(dist).view(-1).to(x.device, x.dtype)
    col = torch.from_numpy(col).view(-1).to(x.device, torch.long)
    row = torch.arange(y.size(0), device=x.device, dtype=torch.long)
    row = row.repeat_interleave(k)
    mask = ~torch.isinf(dist)
    row, col = row[mask], col[mask]

    return torch.stack([row, col], dim=0)


def approx_knn_graph(
    x: Tensor,
    k: int,
    batch: Tensor = None,
    loop: bool = False,
    flow: str = 'source_to_target',
) -> Tensor:  # pragma: no cover
    r"""Computes graph edges to the nearest approximated :obj:`k` points.

    .. note::

        Approximated :math:`k`-nearest neighbor search is performed via the
        `pynndescent <https://pynndescent.readthedocs.io/en/latest>`_ library.

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

    :rtype: :class:`torch.Tensor`
    """
    assert flow in ['source_to_target', 'target_to_source']
    row, col = approx_knn(x, x, k if loop else k + 1, batch, batch)
    row, col = (col, row) if flow == 'source_to_target' else (row, col)
    if not loop:
        mask = row != col
        row, col = row[mask], col[mask]
    return torch.stack([row, col], dim=0)
