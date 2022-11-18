from typing import Any, Optional, Tuple

import torch
from torch import Tensor
from torch_sparse import SparseTensor

from .num_nodes import maybe_num_nodes


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


def is_torch_sparse_tensor(src: Any) -> bool:
    """Returns :obj:`True` if the input :obj:`src` is a PyTorch
    :obj:`SparseTensor` (in any sparse format).

    Args:
        src (Any): The input object to be checked.
    """
    return isinstance(src, Tensor) and src.is_sparse


def is_sparse(src: Any) -> bool:
    """Returns :obj:`True` if the input :obj:`src` is a PyTorch
    :obj:`SparseTensor` (in any sparse format) or a
    :obj:`torch_sparse.SparseTensor`.

    Args:
        src (Any): The input object to be checked.
    """
    return is_torch_sparse_tensor(src) or isinstance(src, SparseTensor)


def to_torch_coo_tensor(
    edge_index: Tensor,
    edge_weight: Optional[Tensor] = None,
    num_nodes: Optional[int] = None,
) -> Tensor:
    """Converts edge index to sparse adjacency matrix
    :class:`torch.sparse.Tensor`.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): The edge weights.
            (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)

    :rtype: :class:`torch.sparse.FloatTensor`

    Example:

        >>> edge_index = torch.tensor([[0, 1, 1, 2, 2, 3],
        ...                            [1, 0, 2, 1, 3, 2]])
        >>> to_torch_coo_tensor(edge_index)
        tensor(indices=tensor([[0, 1, 1, 2, 2, 3],
                            [1, 0, 2, 1, 3, 2]]),
            values=tensor([1., 1., 1., 1., 1., 1.]),
            size=(4, 4), nnz=6, layout=torch.sparse_coo)

    """
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    device = edge_index.device
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), device=device)

    shape = torch.Size((num_nodes, num_nodes))
    adj = torch.sparse_coo_tensor(edge_index, edge_weight, shape,
                                  device=device)
    return adj.coalesce()
