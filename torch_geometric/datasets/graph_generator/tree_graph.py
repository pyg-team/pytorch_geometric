from typing import Optional, Tuple
import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.datasets.graph_generator import GraphGenerator


def tree(depth: int, branch: int, dtype: Optional[torch.dtype] = None, device: Optional[torch.device] = None) -> Tuple[Tensor, Tensor]:
    """
    Generates a tree graph with the given depth and branching factor, along with node positions.

    Args:
        depth (int): The depth of the tree.
        branch (int): The branching factor of the tree.
        dtype (torch.dtype, optional): The desired data type of the returned position tensor. (default: :obj:`None`)
        device (torch.device, optional): The desired device of the returned tensors. (default: :obj:`None`)

    Returns:
        Tuple[Tensor, Tensor]: Edge indices of the tree graph and positions of the nodes.

    Example:
        >>> edge_index, pos = tree(depth=3, branch=2)
        >>> edge_index
        >>> pos
    """

    edges = []
    positions = []
    node_count = 0

    def add_edges(node, current_depth, x, y):
        nonlocal node_count
        if current_depth < depth:
            dx = 1 / (2 ** current_depth)  # Horizontal spacing
            child_y = y - 1  # Move down the tree for each level
            for i in range(branch):
                node_count += 1
                child_x = x + (i - branch / 2 + 0.5) * dx  # Calculate x-position for child
                edges.append((node, node_count))
                positions.append((child_x, child_y))
                add_edges(node_count, current_depth + 1, child_x, child_y)

    # Root node at center-top
    positions.append((0.5, 0))
    add_edges(0, 0, 0.5, 0)

    edge_index = torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()
    pos = torch.tensor(positions, dtype=dtype if dtype is not None else torch.float, device=device)
    return edge_index, pos

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



