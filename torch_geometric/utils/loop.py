from typing import Optional, Tuple, Union

import torch
from torch import Tensor
from torch_scatter import scatter

from torch_geometric.typing import OptTensor

from .num_nodes import maybe_num_nodes


def contains_self_loops(edge_index: Tensor) -> bool:
    r"""Returns :obj:`True` if the graph given by :attr:`edge_index` contains
    self-loops.

    Args:
        edge_index (LongTensor): The edge indices.

    :rtype: bool
    """
    mask = edge_index[0] == edge_index[1]
    return mask.sum().item() > 0


def remove_self_loops(edge_index: Tensor,
                      edge_attr: OptTensor = None) -> Tuple[Tensor, OptTensor]:
    r"""Removes every self-loop in the graph given by :attr:`edge_index`, so
    that :math:`(i,i) \not\in \mathcal{E}` for every :math:`i \in \mathcal{V}`.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional
            edge features. (default: :obj:`None`)

    :rtype: (:class:`LongTensor`, :class:`Tensor`)
    """
    mask = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, mask]
    if edge_attr is None:
        return edge_index, None
    else:
        return edge_index, edge_attr[mask]


def segregate_self_loops(
    edge_index: Tensor, edge_attr: OptTensor = None
) -> Tuple[Tensor, OptTensor, Tensor, OptTensor]:
    r"""Segregates self-loops from the graph.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional
            edge features. (default: :obj:`None`)

    :rtype: (:class:`LongTensor`, :class:`Tensor`, :class:`LongTensor`,
        :class:`Tensor`)
    """

    mask = edge_index[0] != edge_index[1]
    inv_mask = ~mask

    loop_edge_index = edge_index[:, inv_mask]
    loop_edge_attr = None if edge_attr is None else edge_attr[inv_mask]
    edge_index = edge_index[:, mask]
    edge_attr = None if edge_attr is None else edge_attr[mask]

    return edge_index, edge_attr, loop_edge_index, loop_edge_attr


@torch.jit._overload
def add_self_loops(edge_index, edge_attr=None, fill_value=None,
                   num_nodes=None):
    # type: (Tensor, OptTensor, Optional[float], Optional[int]) -> Tuple[Tensor, OptTensor]  # noqa
    pass


# @torch.jit._overload
def add_self_loops(edge_index, edge_attr=None, fill_value=None,
                   num_nodes=None):
    # type: (Tensor, OptTensor, OptTensor, Optional[int]) -> Tuple[Tensor, OptTensor]  # noqa
    pass


@torch.jit._overload
def add_self_loops(edge_index, edge_attr=None, fill_value=None,
                   num_nodes=None):
    # type: (Tensor, OptTensor, Optional[str], Optional[int]) -> Tuple[Tensor, OptTensor]  # noqa
    pass


def add_self_loops(
        edge_index: Tensor, edge_attr: OptTensor = None,
        fill_value: Union[float, Tensor, str] = None,
        num_nodes: Optional[int] = None) -> Tuple[Tensor, OptTensor]:
    r"""Adds a self-loop :math:`(i,i) \in \mathcal{E}` to every node
    :math:`i \in \mathcal{V}` in the graph given by :attr:`edge_index`.
    In case the graph is weighted or has multi-dimensional edge features
    (:obj:`edge_attr != None`), edge features of self-loops will be added
    according to :obj:`fill_value`.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional edge
            features. (default: :obj:`None`)
        fill_value (float or Tensor or str, optional): The way to generate
            edge features of self-loops (in case :obj:`edge_attr != None`).
            If given as :obj:`float` or :class:`torch.Tensor`, edge features of
            self-loops will be directly given by :obj:`fill_value`.
            If given as :obj:`str`, edge features of self-loops are computed by
            aggregating all features of edges that point to the specific node,
            according to a reduce operation. (:obj:`"add"`, :obj:`"mean"`,
            :obj:`"min"`, :obj:`"max"`, :obj:`"mul"`). (default: :obj:`1.`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)

    :rtype: (:class:`LongTensor`, :class:`Tensor`)
    """
    N = maybe_num_nodes(edge_index, num_nodes)

    loop_index = torch.arange(0, N, dtype=torch.long, device=edge_index.device)
    loop_index = loop_index.unsqueeze(0).repeat(2, 1)

    if edge_attr is not None:
        if fill_value is None:
            loop_attr = edge_attr.new_full((N, ) + edge_attr.size()[1:], 1.)

        elif isinstance(fill_value, (int, float)):
            loop_attr = edge_attr.new_full((N, ) + edge_attr.size()[1:],
                                           fill_value)
        elif isinstance(fill_value, Tensor):
            loop_attr = fill_value.to(edge_attr.device, edge_attr.dtype)
            if edge_attr.dim() != loop_attr.dim():
                loop_attr = loop_attr.unsqueeze(0)
            sizes = [N] + [1] * (loop_attr.dim() - 1)
            loop_attr = loop_attr.repeat(*sizes)

        elif isinstance(fill_value, str):
            loop_attr = scatter(edge_attr, edge_index[1], dim=0, dim_size=N,
                                reduce=fill_value)
        else:
            raise AttributeError("No valid 'fill_value' provided")

        edge_attr = torch.cat([edge_attr, loop_attr], dim=0)

    edge_index = torch.cat([edge_index, loop_index], dim=1)
    return edge_index, edge_attr


@torch.jit._overload
def add_remaining_self_loops(edge_index, edge_attr=None, fill_value=None,
                             num_nodes=None):
    # type: (Tensor, OptTensor, Optional[float], Optional[int]) -> Tuple[Tensor, OptTensor]  # noqa
    pass


# @torch.jit._overload
def add_remaining_self_loops(edge_index, edge_attr=None, fill_value=None,
                             num_nodes=None):
    # type: (Tensor, OptTensor, OptTensor, Optional[int]) -> Tuple[Tensor, OptTensor]  # noqa
    pass


@torch.jit._overload
def add_remaining_self_loops(edge_index, edge_attr=None, fill_value=None,
                             num_nodes=None):
    # type: (Tensor, OptTensor, Optional[str], Optional[int]) -> Tuple[Tensor, OptTensor]  # noqa
    pass


def add_remaining_self_loops(
        edge_index: Tensor, edge_attr: OptTensor = None,
        fill_value: Union[float, Tensor, str] = None,
        num_nodes: Optional[int] = None) -> Tuple[Tensor, OptTensor]:
    r"""Adds remaining self-loop :math:`(i,i) \in \mathcal{E}` to every node
    :math:`i \in \mathcal{V}` in the graph given by :attr:`edge_index`.
    In case the graph is weighted or has multi-dimensional edge features
    (:obj:`edge_attr != None`), edge features of non-existing self-loops will
    be added according to :obj:`fill_value`.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional edge
            features. (default: :obj:`None`)
        fill_value (float or Tensor or str, optional): The way to generate
            edge features of self-loops (in case :obj:`edge_attr != None`).
            If given as :obj:`float` or :class:`torch.Tensor`, edge features of
            self-loops will be directly given by :obj:`fill_value`.
            If given as :obj:`str`, edge features of self-loops are computed by
            aggregating all features of edges that point to the specific node,
            according to a reduce operation. (:obj:`"add"`, :obj:`"mean"`,
            :obj:`"min"`, :obj:`"max"`, :obj:`"mul"`). (default: :obj:`1.`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)

    :rtype: (:class:`LongTensor`, :class:`Tensor`)
    """
    N = maybe_num_nodes(edge_index, num_nodes)
    mask = edge_index[0] != edge_index[1]

    loop_index = torch.arange(0, N, dtype=torch.long, device=edge_index.device)
    loop_index = loop_index.unsqueeze(0).repeat(2, 1)

    if edge_attr is not None:
        if fill_value is None:
            loop_attr = edge_attr.new_full((N, ) + edge_attr.size()[1:], 1.)

        elif isinstance(fill_value, (int, float)):
            loop_attr = edge_attr.new_full((N, ) + edge_attr.size()[1:],
                                           fill_value)
        elif isinstance(fill_value, Tensor):
            loop_attr = fill_value.to(edge_attr.device, edge_attr.dtype)
            if edge_attr.dim() != loop_attr.dim():
                loop_attr = loop_attr.unsqueeze(0)
            sizes = [N] + [1] * (loop_attr.dim() - 1)
            loop_attr = loop_attr.repeat(*sizes)

        elif isinstance(fill_value, str):
            loop_attr = scatter(edge_attr, edge_index[1], dim=0, dim_size=N,
                                reduce=fill_value)
        else:
            raise AttributeError("No valid 'fill_value' provided")

        inv_mask = ~mask
        loop_attr[edge_index[0][inv_mask]] = edge_attr[inv_mask]

        edge_attr = torch.cat([edge_attr[mask], loop_attr], dim=0)

    edge_index = torch.cat([edge_index[:, mask], loop_index], dim=1)
    return edge_index, edge_attr
