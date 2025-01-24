from typing import Optional, Tuple

import torch
from torch import Tensor

from torch_geometric.utils import coalesce


def grid(
    height: int,
    width: int,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> Tuple[Tensor, Tensor]:
    r"""Returns the edge indices of a two-dimensional grid graph with height
    :attr:`height` and width :attr:`width` and its node positions.

    Args:
        height (int): The height of the grid.
        width (int): The width of the grid.
        dtype (torch.dtype, optional): The desired data type of the returned
            position tensor. (default: :obj:`None`)
        device (torch.device, optional): The desired device of the returned
            tensors. (default: :obj:`None`)

    :rtype: (:class:`LongTensor`, :class:`Tensor`)

    Example:
        >>> (row, col), pos = grid(height=2, width=2)
        >>> row
        tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])
        >>> col
        tensor([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3])
        >>> pos
        tensor([[0., 1.],
                [1., 1.],
                [0., 0.],
                [1., 0.]])
    """
    edge_index = grid_index(height, width, device)
    pos = grid_pos(height, width, dtype, device)
    return edge_index, pos


def grid_index(
    height: int,
    width: int,
    device: Optional[torch.device] = None,
) -> Tensor:

    w = width
    kernel = torch.tensor(
        [-w - 1, -1, w - 1, -w, 0, w, -w + 1, 1, w + 1],
        device=device,
    )

    row = torch.arange(height * width, dtype=torch.long, device=device)
    row = row.view(-1, 1).repeat(1, kernel.size(0))
    col = row + kernel.view(1, -1)
    row, col = row.view(height, -1), col.view(height, -1)
    index = torch.arange(3, row.size(1) - 3, dtype=torch.long, device=device)
    row, col = row[:, index].view(-1), col[:, index].view(-1)

    mask = (col >= 0) & (col < height * width)
    row, col = row[mask], col[mask]

    edge_index = torch.stack([row, col], dim=0)
    edge_index = coalesce(edge_index, num_nodes=height * width)
    return edge_index


def grid_pos(
    height: int,
    width: int,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> Tensor:

    dtype = torch.float if dtype is None else dtype
    x = torch.arange(width, dtype=dtype, device=device)
    y = (height - 1) - torch.arange(height, dtype=dtype, device=device)

    x = x.repeat(height)
    y = y.unsqueeze(-1).repeat(1, width).view(-1)

    return torch.stack([x, y], dim=-1)
