from typing import Tuple

import torch
from torch import Tensor


def dense_to_sparse(adj: Tensor) -> Tuple[Tensor, Tensor]:
    r"""Converts a dense adjacency matrix to a sparse adjacency matrix defined
    by edge indices and edge attributes.

    Args:
        adj (Tensor): The dense adjacency matrix.
    :rtype: (:class:`LongTensor`, :class:`Tensor`)

    Examples:

        >>> # Forr a single adjacency matrix
        >>> adj = torch.tensor([[3, 1],
        ...                     [2, 0]])
        >>> dense_to_sparse(adj)
        (tensor([[0, 0, 1],
                [0, 1, 0]]),
        tensor([3, 1, 2]))

        >>> # For two adjacency matrixes
        >>> adj = torch.tensor([[[3, 1],
        ...                      [2, 0]],
        ...                     [[0, 1],
        ...                      [0, 2]]])
        >>> dense_to_sparse(adj)
        (tensor([[0, 0, 1, 2, 3],
                [0, 1, 0, 3, 3]]),
        tensor([3, 1, 2, 1, 2]))
    """
    assert adj.dim() >= 2 and adj.dim() <= 3
    assert adj.size(-1) == adj.size(-2)

    edge_index = adj.nonzero().t()

    if edge_index.size(0) == 2:
        edge_attr = adj[edge_index[0], edge_index[1]]
        return edge_index, edge_attr
    else:
        edge_attr = adj[edge_index[0], edge_index[1], edge_index[2]]
        batch = edge_index[0] * adj.size(-1)
        row = batch + edge_index[1]
        col = batch + edge_index[2]
        return torch.stack([row, col], dim=0), edge_attr
