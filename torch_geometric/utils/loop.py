import typing
from typing import Optional, Tuple, Union

import torch
from torch import Tensor

from torch_geometric import EdgeIndex
from torch_geometric.utils import scatter
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils.sparse import (
    is_torch_sparse_tensor,
    to_edge_index,
    to_torch_coo_tensor,
    to_torch_csr_tensor,
)

if typing.TYPE_CHECKING:
    from typing import overload
else:
    from torch.jit import _overload as overload


def contains_self_loops(edge_index: Tensor) -> bool:
    r"""Returns :obj:`True` if the graph given by :attr:`edge_index` contains
    self-loops.

    Args:
        edge_index (LongTensor): The edge indices.

    :rtype: bool

    Examples:
        >>> edge_index = torch.tensor([[0, 1, 0],
        ...                            [1, 0, 0]])
        >>> contains_self_loops(edge_index)
        True

        >>> edge_index = torch.tensor([[0, 1, 1],
        ...                            [1, 0, 2]])
        >>> contains_self_loops(edge_index)
        False
    """
    mask = edge_index[0] == edge_index[1]
    return mask.sum().item() > 0


@overload
def remove_self_loops(
    edge_index: Tensor,
    edge_attr: None = None,
) -> Tuple[Tensor, None]:
    pass


@overload
def remove_self_loops(  # noqa: F811
    edge_index: Tensor,
    edge_attr: Tensor,
) -> Tuple[Tensor, Tensor]:
    pass


@overload
def remove_self_loops(  # noqa: F811
    edge_index: Tensor,
    edge_attr: Optional[Tensor],
) -> Tuple[Tensor, Optional[Tensor]]:
    pass


def remove_self_loops(  # noqa: F811
    edge_index: Tensor,
    edge_attr: Optional[Tensor] = None,
) -> Tuple[Tensor, Optional[Tensor]]:
    r"""Removes every self-loop in the graph given by :attr:`edge_index`, so
    that :math:`(i,i) \not\in \mathcal{E}` for every :math:`i \in \mathcal{V}`.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional
            edge features. (default: :obj:`None`)

    :rtype: (:class:`LongTensor`, :class:`Tensor`)

    Example:
        >>> edge_index = torch.tensor([[0, 1, 0],
        ...                            [1, 0, 0]])
        >>> edge_attr = [[1, 2], [3, 4], [5, 6]]
        >>> edge_attr = torch.tensor(edge_attr)
        >>> remove_self_loops(edge_index, edge_attr)
        (tensor([[0, 1],
                [1, 0]]),
        tensor([[1, 2],
                [3, 4]]))
    """
    size: Optional[Tuple[int, int]] = None
    if not typing.TYPE_CHECKING and torch.jit.is_scripting():
        layout: Optional[int] = None
    else:
        layout: Optional[torch.layout] = None

    value: Optional[Tensor] = None
    if is_torch_sparse_tensor(edge_index):
        layout = edge_index.layout
        size = (edge_index.size(0), edge_index.size(1))
        edge_index, value = to_edge_index(edge_index)

    is_undirected = False
    if not torch.jit.is_scripting() and isinstance(edge_index, EdgeIndex):
        is_undirected = edge_index.is_undirected

    mask = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, mask]

    if not torch.jit.is_scripting() and isinstance(edge_index, EdgeIndex):
        edge_index._is_undirected = is_undirected

    if layout is not None:
        assert edge_attr is None
        assert value is not None
        value = value[mask]
        if str(layout) == 'torch.sparse_coo':  # str(...) for TorchScript :(
            return to_torch_coo_tensor(edge_index, value, size, True), None
        elif str(layout) == 'torch.sparse_csr':
            return to_torch_csr_tensor(edge_index, value, size, True), None
        raise ValueError(f"Unexpected sparse tensor layout (got '{layout}')")

    if edge_attr is None:
        return edge_index, None
    else:
        return edge_index, edge_attr[mask]


@overload
def segregate_self_loops(
    edge_index: Tensor,
    edge_attr: None = None,
) -> Tuple[Tensor, None, Tensor, None]:
    pass


@overload
def segregate_self_loops(  # noqa: F811
    edge_index: Tensor,
    edge_attr: Tensor,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    pass


@overload
def segregate_self_loops(  # noqa: F811
    edge_index: Tensor,
    edge_attr: Optional[Tensor],
) -> Tuple[Tensor, Optional[Tensor], Tensor, Optional[Tensor]]:
    pass


def segregate_self_loops(  # noqa: F811
    edge_index: Tensor,
    edge_attr: Optional[Tensor] = None,
) -> Tuple[Tensor, Optional[Tensor], Tensor, Optional[Tensor]]:
    r"""Segregates self-loops from the graph.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional
            edge features. (default: :obj:`None`)

    :rtype: (:class:`LongTensor`, :class:`Tensor`, :class:`LongTensor`,
        :class:`Tensor`)

    Example:
        >>> edge_index = torch.tensor([[0, 0, 1],
        ...                            [0, 1, 0]])
        >>> (edge_index, edge_attr,
        ...  loop_edge_index,
        ...  loop_edge_attr) = segregate_self_loops(edge_index)
        >>>  loop_edge_index
        tensor([[0],
                [0]])
    """
    mask = edge_index[0] != edge_index[1]
    inv_mask = ~mask

    is_undirected = False
    if not torch.jit.is_scripting() and isinstance(edge_index, EdgeIndex):
        is_undirected = edge_index.is_undirected

    loop_edge_index = edge_index[:, inv_mask]
    loop_edge_attr = None if edge_attr is None else edge_attr[inv_mask]
    edge_index = edge_index[:, mask]
    edge_attr = None if edge_attr is None else edge_attr[mask]

    if not torch.jit.is_scripting() and isinstance(edge_index, EdgeIndex):
        assert isinstance(loop_edge_index, EdgeIndex)
        edge_index._is_undirected = is_undirected
        loop_edge_index._is_undirected = is_undirected

    return edge_index, edge_attr, loop_edge_index, loop_edge_attr


@overload
def add_self_loops(
    edge_index: Tensor,
    edge_attr: None = None,
    fill_value: Optional[float] = None,
    num_nodes: Optional[int] = None,
) -> Tuple[Tensor, None]:
    pass


@overload
def add_self_loops(  # noqa: F811
    edge_index: Tensor,
    edge_attr: None = None,
    fill_value: Optional[float] = None,
    num_nodes: Optional[Tuple[int, int]] = None,
) -> Tuple[Tensor, None]:
    pass


@overload
def add_self_loops(  # noqa: F811
    edge_index: Tensor,
    edge_attr: None = None,
    fill_value: Optional[Tensor] = None,
    num_nodes: Optional[int] = None,
) -> Tuple[Tensor, None]:
    pass


@overload
def add_self_loops(  # noqa: F811
    edge_index: Tensor,
    edge_attr: None = None,
    fill_value: Optional[Tensor] = None,
    num_nodes: Optional[Tuple[int, int]] = None,
) -> Tuple[Tensor, None]:
    pass


@overload
def add_self_loops(  # noqa: F811
    edge_index: Tensor,
    edge_attr: None = None,
    fill_value: Optional[str] = None,
    num_nodes: Optional[int] = None,
) -> Tuple[Tensor, None]:
    pass


@overload
def add_self_loops(  # noqa: F811
    edge_index: Tensor,
    edge_attr: None = None,
    fill_value: Optional[str] = None,
    num_nodes: Optional[Tuple[int, int]] = None,
) -> Tuple[Tensor, None]:
    pass


@overload
def add_self_loops(  # noqa: F811
    edge_index: Tensor,
    edge_attr: Tensor,
    fill_value: Optional[float] = None,
    num_nodes: Optional[int] = None,
) -> Tuple[Tensor, Tensor]:
    pass


@overload
def add_self_loops(  # noqa: F811
    edge_index: Tensor,
    edge_attr: Tensor,
    fill_value: Optional[float] = None,
    num_nodes: Optional[Tuple[int, int]] = None,
) -> Tuple[Tensor, Tensor]:
    pass


@overload
def add_self_loops(  # noqa: F811
    edge_index: Tensor,
    edge_attr: Tensor,
    fill_value: Optional[Tensor] = None,
    num_nodes: Optional[int] = None,
) -> Tuple[Tensor, Tensor]:
    pass


@overload
def add_self_loops(  # noqa: F811
    edge_index: Tensor,
    edge_attr: Tensor,
    fill_value: Optional[Tensor] = None,
    num_nodes: Optional[Tuple[int, int]] = None,
) -> Tuple[Tensor, Tensor]:
    pass


@overload
def add_self_loops(  # noqa: F811
    edge_index: Tensor,
    edge_attr: Tensor,
    fill_value: Optional[str] = None,
    num_nodes: Optional[int] = None,
) -> Tuple[Tensor, Tensor]:
    pass


@overload
def add_self_loops(  # noqa: F811
    edge_index: Tensor,
    edge_attr: Tensor,
    fill_value: Optional[str] = None,
    num_nodes: Optional[Tuple[int, int]] = None,
) -> Tuple[Tensor, Tensor]:
    pass


@overload
def add_self_loops(  # noqa: F811
    edge_index: Tensor,
    edge_attr: Optional[Tensor],
    fill_value: Optional[float] = None,
    num_nodes: Optional[int] = None,
) -> Tuple[Tensor, Optional[Tensor]]:
    pass


@overload
def add_self_loops(  # noqa: F811
    edge_index: Tensor,
    edge_attr: Optional[Tensor],
    fill_value: Optional[float] = None,
    num_nodes: Optional[Tuple[int, int]] = None,
) -> Tuple[Tensor, Optional[Tensor]]:
    pass


@overload
def add_self_loops(  # noqa: F811
    edge_index: Tensor,
    edge_attr: Optional[Tensor],
    fill_value: Optional[Tensor] = None,
    num_nodes: Optional[int] = None,
) -> Tuple[Tensor, Optional[Tensor]]:
    pass


@overload
def add_self_loops(  # noqa: F811
    edge_index: Tensor,
    edge_attr: Optional[Tensor],
    fill_value: Optional[Tensor] = None,
    num_nodes: Optional[Tuple[int, int]] = None,
) -> Tuple[Tensor, Optional[Tensor]]:
    pass


@overload
def add_self_loops(  # noqa: F811
    edge_index: Tensor,
    edge_attr: Optional[Tensor],
    fill_value: Optional[str] = None,
    num_nodes: Optional[int] = None,
) -> Tuple[Tensor, Optional[Tensor]]:
    pass


@overload
def add_self_loops(  # noqa: F811
    edge_index: Tensor,
    edge_attr: Optional[Tensor],
    fill_value: Optional[str] = None,
    num_nodes: Optional[Tuple[int, int]] = None,
) -> Tuple[Tensor, Optional[Tensor]]:
    pass


def add_self_loops(  # noqa: F811
    edge_index: Tensor,
    edge_attr: Optional[Tensor] = None,
    fill_value: Optional[Union[float, Tensor, str]] = None,
    num_nodes: Optional[Union[int, Tuple[int, int]]] = None,
) -> Tuple[Tensor, Optional[Tensor]]:
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
        num_nodes (int or Tuple[int, int], optional): The number of nodes,
            *i.e.* :obj:`max_val + 1` of :attr:`edge_index`.
            If given as a tuple, then :obj:`edge_index` is interpreted as a
            bipartite graph with shape :obj:`(num_src_nodes, num_dst_nodes)`.
            (default: :obj:`None`)

    :rtype: (:class:`LongTensor`, :class:`Tensor`)

    Examples:
        >>> edge_index = torch.tensor([[0, 1, 0],
        ...                            [1, 0, 0]])
        >>> edge_weight = torch.tensor([0.5, 0.5, 0.5])
        >>> add_self_loops(edge_index)
        (tensor([[0, 1, 0, 0, 1],
                [1, 0, 0, 0, 1]]),
        None)

        >>> add_self_loops(edge_index, edge_weight)
        (tensor([[0, 1, 0, 0, 1],
                [1, 0, 0, 0, 1]]),
        tensor([0.5000, 0.5000, 0.5000, 1.0000, 1.0000]))

        >>> # edge features of self-loops are filled by constant `2.0`
        >>> add_self_loops(edge_index, edge_weight,
        ...                fill_value=2.)
        (tensor([[0, 1, 0, 0, 1],
                [1, 0, 0, 0, 1]]),
        tensor([0.5000, 0.5000, 0.5000, 2.0000, 2.0000]))

        >>> # Use 'add' operation to merge edge features for self-loops
        >>> add_self_loops(edge_index, edge_weight,
        ...                fill_value='add')
        (tensor([[0, 1, 0, 0, 1],
                [1, 0, 0, 0, 1]]),
        tensor([0.5000, 0.5000, 0.5000, 1.0000, 0.5000]))
    """
    if not typing.TYPE_CHECKING and torch.jit.is_scripting():
        layout: Optional[int] = None
    else:
        layout: Optional[torch.layout] = None
    is_sparse = is_torch_sparse_tensor(edge_index)

    value: Optional[Tensor] = None
    if is_sparse:
        assert edge_attr is None
        layout = edge_index.layout
        size = (edge_index.size(0), edge_index.size(1))
        N = min(size)
        edge_index, value = to_edge_index(edge_index)
    elif isinstance(num_nodes, (tuple, list)):
        size = (num_nodes[0], num_nodes[1])
        N = min(size)
    else:
        N = maybe_num_nodes(edge_index, num_nodes)
        size = (N, N)

    device = edge_index.device
    if torch.jit.is_scripting():
        loop_index = torch.arange(0, N, device=device).view(1, -1).repeat(2, 1)
    else:
        loop_index = EdgeIndex(
            torch.arange(0, N, device=device).view(1, -1).repeat(2, 1),
            sparse_size=(N, N),
            is_undirected=True,
        )
    full_edge_index = torch.cat([edge_index, loop_index], dim=1)

    if is_sparse:
        assert edge_attr is None
        assert value is not None
        loop_attr = compute_loop_attr(  #
            edge_index, value, N, is_sparse, fill_value)
        value = torch.cat([value, loop_attr], dim=0)

        if str(layout) == 'torch.sparse_coo':  # str(...) for TorchScript :(
            return to_torch_coo_tensor(full_edge_index, value, size), None
        elif str(layout) == 'torch.sparse_csr':
            return to_torch_csr_tensor(full_edge_index, value, size), None
        raise ValueError(f"Unexpected sparse tensor layout (got '{layout}')")

    if edge_attr is not None:
        loop_attr = compute_loop_attr(  #
            edge_index, edge_attr, N, is_sparse, fill_value)
        edge_attr = torch.cat([edge_attr, loop_attr], dim=0)

    return full_edge_index, edge_attr


@overload
def add_remaining_self_loops(
    edge_index: Tensor,
    edge_attr: None = None,
    fill_value: Optional[float] = None,
    num_nodes: Optional[int] = None,
) -> Tuple[Tensor, None]:
    pass


@overload
def add_remaining_self_loops(  # noqa: F811
    edge_index: Tensor,
    edge_attr: None = None,
    fill_value: Optional[Tensor] = None,
    num_nodes: Optional[int] = None,
) -> Tuple[Tensor, None]:
    pass


@overload
def add_remaining_self_loops(  # noqa: F811
    edge_index: Tensor,
    edge_attr: None = None,
    fill_value: Optional[str] = None,
    num_nodes: Optional[int] = None,
) -> Tuple[Tensor, None]:
    pass


@overload
def add_remaining_self_loops(  # noqa: F811
    edge_index: Tensor,
    edge_attr: Tensor,
    fill_value: Optional[float] = None,
    num_nodes: Optional[int] = None,
) -> Tuple[Tensor, Tensor]:
    pass


@overload
def add_remaining_self_loops(  # noqa: F811
    edge_index: Tensor,
    edge_attr: Tensor,
    fill_value: Optional[Tensor] = None,
    num_nodes: Optional[int] = None,
) -> Tuple[Tensor, Tensor]:
    pass


@overload
def add_remaining_self_loops(  # noqa: F811
    edge_index: Tensor,
    edge_attr: Tensor,
    fill_value: Optional[str] = None,
    num_nodes: Optional[int] = None,
) -> Tuple[Tensor, Tensor]:
    pass


@overload
def add_remaining_self_loops(  # noqa: F811
    edge_index: Tensor,
    edge_attr: Optional[Tensor],
    fill_value: Optional[float] = None,
    num_nodes: Optional[int] = None,
) -> Tuple[Tensor, Optional[Tensor]]:
    pass


@overload
def add_remaining_self_loops(  # noqa: F811
    edge_index: Tensor,
    edge_attr: Optional[Tensor],
    fill_value: Optional[Tensor] = None,
    num_nodes: Optional[int] = None,
) -> Tuple[Tensor, Optional[Tensor]]:
    pass


@overload
def add_remaining_self_loops(  # noqa: F811
    edge_index: Tensor,
    edge_attr: Optional[Tensor],
    fill_value: Optional[str] = None,
    num_nodes: Optional[int] = None,
) -> Tuple[Tensor, Optional[Tensor]]:
    pass


def add_remaining_self_loops(  # noqa: F811
    edge_index: Tensor,
    edge_attr: Optional[Tensor] = None,
    fill_value: Optional[Union[float, Tensor, str]] = None,
    num_nodes: Optional[int] = None,
) -> Tuple[Tensor, Optional[Tensor]]:
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

    Example:
        >>> edge_index = torch.tensor([[0, 1],
        ...                            [1, 0]])
        >>> edge_weight = torch.tensor([0.5, 0.5])
        >>> add_remaining_self_loops(edge_index, edge_weight)
        (tensor([[0, 1, 0, 1],
                [1, 0, 0, 1]]),
        tensor([0.5000, 0.5000, 1.0000, 1.0000]))
    """
    N = maybe_num_nodes(edge_index, num_nodes)
    mask = edge_index[0] != edge_index[1]

    device = edge_index.device
    if torch.jit.is_scripting():
        loop_index = torch.arange(0, N, device=device).view(1, -1).repeat(2, 1)
    else:
        loop_index = EdgeIndex(
            torch.arange(0, N, device=device).view(1, -1).repeat(2, 1),
            sparse_size=(N, N),
            is_undirected=True,
        )

    if edge_attr is not None:

        loop_attr = compute_loop_attr(  #
            edge_index, edge_attr, N, False, fill_value)

        inv_mask = ~mask
        loop_attr[edge_index[0][inv_mask]] = edge_attr[inv_mask]

        edge_attr = torch.cat([edge_attr[mask], loop_attr], dim=0)

    is_undirected = False
    if not torch.jit.is_scripting() and isinstance(edge_index, EdgeIndex):
        is_undirected = edge_index.is_undirected

    edge_index = edge_index[:, mask]

    if not torch.jit.is_scripting() and isinstance(edge_index, EdgeIndex):
        edge_index._is_undirected = is_undirected

    edge_index = torch.cat([edge_index, loop_index], dim=1)

    return edge_index, edge_attr


def get_self_loop_attr(
    edge_index: Tensor,
    edge_attr: Optional[Tensor] = None,
    num_nodes: Optional[int] = None,
) -> Tensor:
    r"""Returns the edge features or weights of self-loops
    :math:`(i, i)` of every node :math:`i \in \mathcal{V}` in the
    graph given by :attr:`edge_index`. Edge features of missing self-loops not
    present in :attr:`edge_index` will be filled with zeros. If
    :attr:`edge_attr` is not given, it will be the vector of ones.

    .. note::
        This operation is analogous to getting the diagonal elements of the
        dense adjacency matrix.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional edge
            features. (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)

    :rtype: :class:`Tensor`

    Examples:
        >>> edge_index = torch.tensor([[0, 1, 0],
        ...                            [1, 0, 0]])
        >>> edge_weight = torch.tensor([0.2, 0.3, 0.5])
        >>> get_self_loop_attr(edge_index, edge_weight)
        tensor([0.5000, 0.0000])

        >>> get_self_loop_attr(edge_index, edge_weight, num_nodes=4)
        tensor([0.5000, 0.0000, 0.0000, 0.0000])
    """
    loop_mask = edge_index[0] == edge_index[1]
    loop_index = edge_index[0][loop_mask]

    if edge_attr is not None:
        loop_attr = edge_attr[loop_mask]
    else:  # A vector of ones:
        loop_attr = torch.ones(loop_index.numel(), device=edge_index.device)

    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    full_loop_attr = loop_attr.new_zeros((num_nodes, ) + loop_attr.size()[1:])
    full_loop_attr[loop_index] = loop_attr

    return full_loop_attr


@overload
def compute_loop_attr(
    edge_index: Tensor,
    edge_attr: Tensor,
    num_nodes: int,
    is_sparse: bool,
    fill_value: Optional[float] = None,
) -> Tensor:
    pass


@overload
def compute_loop_attr(  # noqa: F811
    edge_index: Tensor,
    edge_attr: Tensor,
    num_nodes: int,
    is_sparse: bool,
    fill_value: Optional[Tensor] = None,
) -> Tensor:
    pass


@overload
def compute_loop_attr(  # noqa: F811
    edge_index: Tensor,
    edge_attr: Tensor,
    num_nodes: int,
    is_sparse: bool,
    fill_value: Optional[str] = None,
) -> Tensor:
    pass


def compute_loop_attr(  # noqa: F811
    edge_index: Tensor,
    edge_attr: Tensor,
    num_nodes: int,
    is_sparse: bool,
    fill_value: Optional[Union[float, Tensor, str]] = None,
) -> Tensor:

    if fill_value is None:
        size = (num_nodes, ) + edge_attr.size()[1:]
        return edge_attr.new_ones(size)

    elif isinstance(fill_value, (int, float)):
        size = (num_nodes, ) + edge_attr.size()[1:]
        return edge_attr.new_full(size, fill_value)

    elif isinstance(fill_value, Tensor):
        size = (num_nodes, ) + edge_attr.size()[1:]
        loop_attr = fill_value.to(edge_attr.device, edge_attr.dtype)
        if edge_attr.dim() != loop_attr.dim():
            loop_attr = loop_attr.unsqueeze(0)
        return loop_attr.expand(size).contiguous()

    elif isinstance(fill_value, str):
        col = edge_index[0] if is_sparse else edge_index[1]
        return scatter(edge_attr, col, 0, num_nodes, fill_value)

    raise AttributeError("No valid 'fill_value' provided")
