from typing import Optional

import torch

from torch_geometric.data import Data
from torch_geometric.datasets.graph_generator import GraphGenerator
from torch_geometric.utils import grid


class GridGraph(GraphGenerator):
    r"""Generates two-dimensional grid graphs.
    See :meth:`~torch_geometric.utils.grid` for more information.

    Args:
        height (int): The height of the grid.
        width (int): The width of the grid.
        dtype (:obj:`torch.dtype`, optional): The desired data type of the
            returned position tensor. (default: :obj:`None`)
    """
    def __init__(
        self,
        height: int,
        width: int,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.height = height
        self.width = width
        self.dtype = dtype

    def __call__(self) -> Data:
        edge_index, pos = grid(height=self.height, width=self.width,
                               dtype=self.dtype)
        return Data(edge_index=edge_index, pos=pos)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(height={self.height}, '
                f'width={self.width})')
