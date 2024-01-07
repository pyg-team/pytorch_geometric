from typing import Optional

import torch

from torch_geometric.data import Data
from torch_geometric.datasets.graph_generator import GraphGenerator
from torch_geometric.utils import tree


class TreeGraph(GraphGenerator):
    r"""Generates two-dimensional grid graphs.
    See :meth:`~torch_geometric.utils.grid` for more information.

    Args:
        depth (int): The depth of the tree.
        branck (int): The branch of the tree.
        dtype (:obj:`torch.dtype`, optional): The desired data type of the
            returned position tensor. (default: :obj:`None`)
    """
    def __init__(
        self,
        depth: int,
        branch: int,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.depth = depth
        self.branch = branch
        self.dtype = dtype

    def __call__(self) -> Data:
        edge_index, pos = tree(depth=self.depth, branch=self.branch, dtype=self.dtype)
        return Data(edge_index=edge_index, pos=pos)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(depth={self.depth}, '
                f'branch={self.branch})')
