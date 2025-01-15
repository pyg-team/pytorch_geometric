from typing import Optional, Tuple, Union

import torch
from torch import Tensor

from torch_geometric.utils import remove_self_loops, segregate_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes


def num_isolated_nodes(
    edge_index: Tensor,
    num_nodes: Optional[Union[int, Tuple[int, int]]] = None,
) -> Union[int, Tuple[int, int]]:
    r"""Returns the number of isolated nodes in the graph given by
    :attr:`edge_index`. Please specify :obj:`num_nodes` as a tuple of two
    integers if the graph is bipartite. For normal graph, self loops are
    automatically removed before checking for isolated nodes.

    Args:
        edge_index (LongTensor): The edge indices.
        num_nodes (int or tuple, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. For bipartite graph,
            provide a tuple of two integers `(num_src_nodes, num_dst_nodes)`.
            (default: :obj:`None`)

    :rtype: int or tuple

    Examples:
        >>> edge_index = torch.tensor([[0, 1, 0],
        ...                            [1, 0, 0]])
        >>> num_isolated_nodes(edge_index)
        0
        >>> bi_edge_index = torch.tensor([[0, 1],
        ...                               [1, 0]])
        >>> num_isolated_nodes(bi_edge_index, num_nodes=(2, 2))
        (0, 0)
        >>> num_isolated_nodes(bi_edge_index, num_nodes=(2, 3))
        (0, 1)
        >>> num_isolated_nodes(bi_edge_index, num_nodes=(3, 3))
        (1, 1)
    """
    if not isinstance(num_nodes, tuple):
        num_nodes = maybe_num_nodes(edge_index, num_nodes)
        edge_index, _ = remove_self_loops(edge_index)
        return num_nodes - torch.unique(edge_index.view(-1)).numel()
    else:
        num_src_nodes, num_dst_nodes = num_nodes
        num_src_isolated = num_src_nodes - torch.unique(
            edge_index[0, :]).numel()
        num_dst_isolated = num_dst_nodes - torch.unique(
            edge_index[1, :]).numel()
    return (num_src_isolated, num_dst_isolated)


def contains_isolated_nodes(
    edge_index: Tensor,
    num_nodes: Optional[Union[int, Tuple[int, int]]] = None,
) -> Union[bool, Tuple[bool, bool]]:
    r"""For normal graph, returns :obj:`True` if the graph given by
    :attr:`edge_index` contains isolated nodes and self loops are
    automatically removed before checking for isolated nodes.
    For bipartite graph, please specify :obj:`num_nodes` as a tuple
    of two integers and returns a tuple of two booleans where the first
    boolean indicates whether the source nodes contain isolated nodes
    and the second boolean indicates whether the destination nodes
    contain isolated nodes.

    Args:
        edge_index (LongTensor): The edge indices.
        num_nodes (int or tuple, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. For bipartite graph,
            provide a tuple of two integers `(num_src_nodes, num_dst_nodes)`.
            (default: :obj:`None`)

    :rtype: bool or tuple

    Examples:
        >>> edge_index = torch.tensor([[0, 1, 0],
        ...                            [1, 0, 0]])
        >>> contains_isolated_nodes(edge_index)
        False

        >>> contains_isolated_nodes(edge_index, num_nodes=3)
        True

        >>> bi_edge_index = torch.tensor([[0, 0],
        ...                               [1, 1]])
        >>> contains_isolated_nodes(bi_edge_index, num_nodes=(2, 2))
        (False, False)
        >>> contains_isolated_nodes(bi_edge_index, num_nodes=(2, 3))
        (False, True)
        >>> contains_isolated_nodes(bi_edge_index, num_nodes=(3, 3))
        (True, True)
    """
    if not isinstance(num_nodes, tuple):
        num_nodes = maybe_num_nodes(edge_index, num_nodes)
        edge_index, _ = remove_self_loops(edge_index)
        return torch.unique(edge_index.view(-1)).numel() < num_nodes
    else:
        num_src_nodes, num_dst_nodes = num_nodes
        num_nodes = num_src_nodes + num_dst_nodes
        num_src_isolated = num_src_nodes - torch.unique(
            edge_index[0, :]).numel()
        num_dst_isolated = num_dst_nodes - torch.unique(
            edge_index[1, :]).numel()
        return (num_src_isolated > 0, num_dst_isolated > 0)


def remove_isolated_nodes(
    edge_index: Tensor,
    edge_attr: Optional[Tensor] = None,
    num_nodes: Optional[Union[int, Tuple[int, int]]] = None,
) -> Tuple[Tensor, Optional[Tensor], Union[Tensor, Tuple[Tensor, Tensor]]]:
    r"""Removes the isolated nodes from the graph given by :attr:`edge_index`
    with optional edge attributes :attr:`edge_attr`.
    In addition, returns a mask of shape :obj:`[num_nodes]` to manually filter
    out isolated node features later on.
    Self-loops are preserved for non-isolated nodes.
    Please specify :obj:`num_nodes` as a tuple of two integers
    if the graph is bipartite.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional
            edge features. (default: :obj:`None`)
        num_nodes (int or tuple, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`.For bipartite graph,
            provide a tuple of two integers `(num_src_nodes, num_dst_nodes)`.
            (default: :obj:`None`)

    :rtype: (LongTensor, Tensor, BoolTensor or Tuple[BoolTensor, BoolTensor])

    Examples:
        >>> edge_index = torch.tensor([[0, 1, 0],
        ...                            [1, 0, 0]])
        >>> edge_index, edge_attr, mask = remove_isolated_nodes(edge_index)
        >>> mask # node mask (2 nodes)
        tensor([True, True])

        >>> edge_index, edge_attr, mask = remove_isolated_nodes(edge_index,
        ...                                                     num_nodes=3)
        >>> mask # node mask (3 nodes)
        tensor([True, True, False])

        >>> bi_edge_index = torch.tensor([[0, 1],
        ...                               [1, 2]])
        >>> edge_index, _, (src_mask, dst_mask) = \
        ... remove_isolated_nodes(bi_edge_index, num_nodes=(2, 3))
        >>> src_mask # source node mask (2 nodes)
        tensor([True, True])
        >>> dst_mask # destination node mask (3 nodes)
        tensor([False, True, True])
    """
    if not isinstance(num_nodes, tuple):
        num_nodes = maybe_num_nodes(edge_index, num_nodes)
        out = segregate_self_loops(edge_index, edge_attr)
        edge_index, edge_attr, loop_edge_index, loop_edge_attr = out
        mask = torch.zeros(num_nodes, dtype=torch.bool,
                           device=edge_index.device)
        mask[edge_index.view(-1)] = 1
        assoc = torch.full((num_nodes, ), -1, dtype=torch.long,
                           device=mask.device)
        assoc[mask] = torch.arange(mask.sum(),
                                   device=assoc.device)  # type: ignore
        edge_index = assoc[edge_index]

        loop_mask = torch.zeros_like(mask)
        loop_mask[loop_edge_index[0]] = 1
        loop_mask = loop_mask & mask
        loop_assoc = torch.full_like(assoc, -1)
        loop_assoc[loop_edge_index[0]] = torch.arange(loop_edge_index.size(1),
                                                      device=loop_assoc.device)
        loop_idx = loop_assoc[loop_mask]
        loop_edge_index = assoc[loop_edge_index[:, loop_idx]]
        edge_index = torch.cat([edge_index, loop_edge_index], dim=1)

        if edge_attr is not None:
            assert loop_edge_attr is not None
            loop_edge_attr = loop_edge_attr[loop_idx]
            edge_attr = torch.cat([edge_attr, loop_edge_attr], dim=0)
        return edge_index, edge_attr, mask
    else:
        num_src_nodes, num_dst_nodes = num_nodes
        src_mask = torch.zeros(num_src_nodes, dtype=torch.bool,
                               device=edge_index.device)
        src_mask[edge_index[0, :]] = 1
        src_assoc = torch.full((num_src_nodes, ), -1, dtype=torch.long,
                               device=src_mask.device)
        src_assoc[src_mask] = torch.arange(
            src_mask.sum(), device=src_assoc.device)  # type: ignore
        src_edge_index = src_assoc[edge_index[0, :]]

        dst_mask = torch.zeros(num_dst_nodes, dtype=torch.bool,
                               device=edge_index.device)
        dst_mask[edge_index[1, :]] = 1
        dst_assoc = torch.full((num_dst_nodes, ), -1, dtype=torch.long,
                               device=dst_mask.device)
        dst_assoc[dst_mask] = torch.arange(
            dst_mask.sum(), device=dst_assoc.device)  # type: ignore
        dst_edge_index = dst_assoc[edge_index[1, :]]
        edge_index = torch.stack([src_edge_index, dst_edge_index], dim=0)
        return edge_index, edge_attr, (src_mask, dst_mask)
