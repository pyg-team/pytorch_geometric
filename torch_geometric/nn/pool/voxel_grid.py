from typing import List, Optional, Union

import torch
from torch import Tensor

from torch_geometric.utils.repeat import repeat

try:
    from torch_cluster import grid_cluster
except ImportError:
    grid_cluster = None


def voxel_grid(
    pos: Tensor,
    size: Union[float, List[float], Tensor],
    batch: Optional[Tensor] = None,
    start: Optional[Union[float, List[float], Tensor]] = None,
    end: Optional[Union[float, List[float], Tensor]] = None,
) -> Tensor:
    r"""Voxel grid pooling from the, *e.g.*, `Dynamic Edge-Conditioned Filters
    in Convolutional Networks on Graphs <https://arxiv.org/abs/1704.02901>`_
    paper, which overlays a regular grid of user-defined size over a point
    cloud and clusters all points within the same voxel.

    Args:
        pos (torch.Tensor): Node position matrix
            :math:`\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times D}`.
        size (float or [float] or Tensor): Size of a voxel (in each dimension).
        batch (torch.Tensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots,B-1\}}^N`, which assigns each
            node to a specific example. (default: :obj:`None`)
        start (float or [float] or Tensor, optional): Start coordinates of the
            grid (in each dimension). If set to :obj:`None`, will be set to the
            minimum coordinates found in :attr:`pos`. (default: :obj:`None`)
        end (float or [float] or Tensor, optional): End coordinates of the grid
            (in each dimension). If set to :obj:`None`, will be set to the
            maximum coordinates found in :attr:`pos`. (default: :obj:`None`)

    :rtype: :class:`torch.Tensor`
    """

    if grid_cluster is None:
        raise ImportError('`voxel_grid` requires `torch-cluster`.')

    pos = pos.unsqueeze(-1) if pos.dim() == 1 else pos
    dim = pos.size(1)

    if batch is None:
        batch = pos.new_zeros(pos.size(0), dtype=torch.long)

    pos = torch.cat([pos, batch.view(-1, 1).to(pos.dtype)], dim=-1)

    if not isinstance(size, Tensor):
        size = torch.tensor(size, dtype=pos.dtype, device=pos.device)
    size = repeat(size, dim)
    size = torch.cat([size, size.new_ones(1)])  # Add additional batch dim.

    if start is not None:
        if not isinstance(start, Tensor):
            start = torch.tensor(start, dtype=pos.dtype, device=pos.device)
        start = repeat(start, dim)
        start = torch.cat([start, start.new_zeros(1)])

    if end is not None:
        if not isinstance(end, Tensor):
            end = torch.tensor(end, dtype=pos.dtype, device=pos.device)
        end = repeat(end, dim)
        end = torch.cat([end, batch.max().unsqueeze(0)])

    return grid_cluster(pos, size, start, end)
