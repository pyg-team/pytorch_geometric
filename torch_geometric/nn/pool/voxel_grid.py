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
    num_nodes, dim = pos.size()

    size = size.tolist() if torch.is_tensor(size) else size
    start = start.tolist() if torch.is_tensor(start) else start
    end = end.tolist() if torch.is_tensor(end) else end

    size, start, end = repeat(size, dim), repeat(start, dim), repeat(end, dim)

    if batch is None:
        batch = torch.zeros(pos.shape[0], dtype=torch.long)

    pos = torch.cat([pos, batch.unsqueeze(-1).type_as(pos)], dim=-1)
    size = size + [1]
    start = None if start is None else start + [0]
    end = None if end is None else end + [batch.max().item()]

    size = torch.tensor(size, dtype=pos.dtype, device=pos.device)
    if start is not None:
        start = torch.tensor(start, dtype=pos.dtype, device=pos.device)
    if end is not None:
        end = torch.tensor(end, dtype=pos.dtype, device=pos.device)

    return grid_cluster(pos, size, start, end)
