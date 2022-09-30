from typing import Optional, Tuple, Union

import torch
from torch import Tensor

from .num_nodes import maybe_num_nodes


def shuffle_node(x: Tensor, training: bool = True) -> Tuple[Tensor, Tensor]:
    r"""Randomly shuffle the feature matrix :obj:`x` along the
    first dimmension.

    The method returns (1) the shuffled :obj:`x`, (2) the permutation
    indicating the orders of original nodes after shuffling.

    Args:
        x (FloatTensor): The feature matrix.
        training (bool, optional): If set to :obj:`False`, this operation is a
            no-op. (default: :obj:`True`)

    :rtype: (:class:`FloatTensor`, :class:`LongTensor`)

    Example:

        >>> x = torch.tensor([[0, 1, 2],
        ...                   [3, 4, 5],
        ...                   [6, 7, 8]], dtype=torch.float)
        >>> x, node_perm = shuffle_node(x)
        >>> x
        tensor([[6., 7., 8.],
                [3., 4., 5.],
                [0., 1., 2.]])
        >>> node_perm
        tensor([2, 1, 0])
    """
    if not training:
        perm = torch.arange(x.size(0), device=x.device)
        return x, perm
    perm = torch.randperm(x.size(0), device=x.device)
    return x[perm], perm


def mask_feature(x: Tensor, p: float = 0.5, column_wise: bool = True,
                 training: bool = True) -> Tuple[Tensor, Tensor]:
    r"""Randomly masks feature from the feature matrix
    :obj:`x` with probability :obj:`p` using samples from
    a Bernoulli distribution.

    The method returns (1) the retained :obj:`x`, (2) the feature
    mask with the same shape as :obj:`x`, indicating where features are retained.

    Args:
        x (FloatTensor): The feature matrix.
        p (float, optional): The masking ratio. (default: :obj:`0.5`)
        column_wise (bool, optional): If set to :obj:`True`, masks whole feature
             columns with probability :obj:`p`.  (default: :obj:`True`)
        training (bool, optional): If set to :obj:`False`, this operation is a
            no-op. (default: :obj:`True`)

    :rtype: (:class:`FloatTensor`, :class:`BoolTensor`)

    Examples:

        >>> # Masked features are column-wise sampled
        >>> x = torch.tensor([[0, 1, 2],
        ...                   [3, 4, 5],
        ...                   [6, 7, 8]], dtype=torch.float)
        >>> x, feat_mask = mask_feature(x)
        >>> x
        tensor([[0., 0., 2.],
                [3., 0., 5.],
                [6., 0., 8.]]),
        >>> feat_mask
        tensor([[ True, False,  True],
                [ True, False,  True],
                [ True, False,  True]])

        >>> # Masked features are uniformly sampled
        >>> x, feat_mask = mask_feature(x, column_wise=False)
        >>> x
        tensor([[0., 0., 0.],
                [3., 0., 5.],
                [0., 0., 8.]])
        >>> feat_mask
        tensor([[False, False, False],
                [ True, False,  True],
                [False, False,  True]])
    """
    if p < 0. or p > 1.:
        raise ValueError(f'Masking ratio has to be between 0 and 1 '
                         f'(got {p}')
    if not training or p == 0.0:
        return x, torch.ones_like(x, dtype=torch.bool)

    if column_wise:
        mask = torch.rand(x.size(1), device=x.device) >= p
        mask = mask.tile(x.size(0), 1)
    else:
        mask = torch.randn_like(x) >= p

    # To avoid inplace operation
    x = x * (mask.float())
    return x, mask


def add_random_edge(edge_index, p: float = 0.2, force_undirected: bool = False,
                    num_nodes: Optional[Union[Tuple[int], int]] = None,
                    training: bool = True) -> Tuple[Tensor, Tensor]:
    r"""Randomly adds edges to :obj:`edge_index`.

    The method returns (1) the retained :obj:`edge_index`, (2) the added
    edge indices.

    Args:
        edge_index (LongTensor): The edge indices.
        p (float, optional): Ratio of added edges to the existing edges.
            (default: :obj:`0.2`)
        force_undirected (bool, optional): If set to :obj:`True`,
            added edges will be undirected.
            (default: :obj:`False`)
        num_nodes (int, Tuple[int], optional): The overall number of nodes,
            *i.e.* :obj:`max_val + 1`, or the number of source and
            destination nodes, *i.e.* :obj:`(max_src_val + 1, max_dst_val + 1)`
            of :attr:`edge_index`. (default: :obj:`None`)
        training (bool, optional): If set to :obj:`False`, this operation is a
            no-op. (default: :obj:`True`)

    :rtype: (:class:`LongTensor`, :class:`LongTensor`)

    Examples:

        >>> # Standard case
        >>> edge_index = torch.tensor([[0, 1, 1, 2, 2, 3],
        ...                            [1, 0, 2, 1, 3, 2]])
        >>> edge_index, added_edges = add_random_edge(edge_index, p=0.5)
        >>> edge_index
        tensor([[0, 1, 1, 2, 2, 3, 2, 1, 3],
                [1, 0, 2, 1, 3, 2, 0, 2, 1]])
        >>> added_edges
        tensor([[2, 1, 3],
                [0, 2, 1]])

        >>> # The returned graph is kept undirected
        >>> edge_index, added_edges = add_random_edge(edge_index, p=0.5,
        ...                                           force_undirected=True)
        >>> edge_index
        tensor([[0, 1, 1, 2, 2, 3, 2, 1, 3, 0, 2, 1],
                [1, 0, 2, 1, 3, 2, 0, 2, 1, 2, 1, 3]])
        >>> added_edges
        tensor([[2, 1, 3, 0, 2, 1],
                [0, 2, 1, 2, 1, 3]])

        >>> # For bipartite graphs
        >>> edge_index = torch.tensor([[0, 1, 2, 3, 4, 5],
        ...                            [2, 3, 1, 4, 2, 1]])
        >>> edge_index, added_edges = add_random_edge(edge_index, p=0.5,
        ...                                           num_nodes=(6, 5))
        >>> edge_index
        tensor([[0, 1, 2, 3, 4, 5, 3, 4, 1],
                [2, 3, 1, 4, 2, 1, 1, 3, 2]])
        >>> added_edges
        tensor([[3, 4, 1],
                [1, 3, 2]])
    """
    if p < 0. or p > 1.:
        raise ValueError(f'Ratio of added edges has to be between 0 and 1 '
                         f'(got {p}')

    device = edge_index.device
    if not training or p == 0.0:
        edge_index_to_add = torch.tensor([], device=device).view(2, 0)
        return edge_index, edge_index_to_add

    num_nodes = (num_nodes,
                 num_nodes) if not isinstance(num_nodes, tuple) else num_nodes
    num_src_nodes = maybe_num_nodes(edge_index, num_nodes[0])
    num_dst_nodes = maybe_num_nodes(edge_index, num_nodes[1])

    num_edges_to_add = round(edge_index.size(1) * p)
    row = torch.randint(0, num_src_nodes, size=(num_edges_to_add, ))
    col = torch.randint(0, num_dst_nodes, size=(num_edges_to_add, ))

    if force_undirected:
        edge_index_to_add = torch.stack(
            [torch.cat([row, col], dim=0),
             torch.cat([col, row], dim=0)], dim=0).to(device)
    else:
        edge_index_to_add = torch.stack([row, col], dim=0).to(device)
    edge_index = torch.cat([edge_index, edge_index_to_add], dim=1)
    return edge_index, edge_index_to_add
