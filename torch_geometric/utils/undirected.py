import typing
from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor

from torch_geometric.typing import OptTensor
from torch_geometric.utils import coalesce, sort_edge_index
from torch_geometric.utils.num_nodes import maybe_num_nodes

if typing.TYPE_CHECKING:
    from typing import overload
else:
    from torch.jit import _overload as overload

MISSING = '???'


@overload
def is_undirected(
    edge_index: Tensor,
    edge_attr: Optional[Tensor] = None,
    num_nodes: Optional[int] = None,
) -> bool:
    pass


@overload
def is_undirected(  # noqa: F811
    edge_index: Tensor,
    edge_attr: List[Tensor],
    num_nodes: Optional[int] = None,
) -> bool:
    pass


def is_undirected(  # noqa: F811
    edge_index: Tensor,
    edge_attr: Union[Optional[Tensor], List[Tensor]] = None,
    num_nodes: Optional[int] = None,
) -> bool:
    r"""Returns :obj:`True` if the graph given by :attr:`edge_index` is
    undirected.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor or List[Tensor], optional): Edge weights or multi-
            dimensional edge features.
            If given as a list, will check for equivalence in all its entries.
            (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max(edge_index) + 1`. (default: :obj:`None`)

    :rtype: bool

    Examples:
        >>> edge_index = torch.tensor([[0, 1, 0],
        ...                         [1, 0, 0]])
        >>> weight = torch.tensor([0, 0, 1])
        >>> is_undirected(edge_index, weight)
        True

        >>> weight = torch.tensor([0, 1, 1])
        >>> is_undirected(edge_index, weight)
        False

    """
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    edge_attrs: List[Tensor] = []
    if isinstance(edge_attr, Tensor):
        edge_attrs.append(edge_attr)
    elif isinstance(edge_attr, (list, tuple)):
        edge_attrs = edge_attr

    edge_index1, edge_attrs1 = sort_edge_index(
        edge_index,
        edge_attrs,
        num_nodes=num_nodes,
        sort_by_row=True,
    )
    edge_index2, edge_attrs2 = sort_edge_index(
        edge_index,
        edge_attrs,
        num_nodes=num_nodes,
        sort_by_row=False,
    )

    if not torch.equal(edge_index1[0], edge_index2[1]):
        return False

    if not torch.equal(edge_index1[1], edge_index2[0]):
        return False

    assert isinstance(edge_attrs1, list) and isinstance(edge_attrs2, list)
    for edge_attr1, edge_attr2 in zip(edge_attrs1, edge_attrs2):
        if not torch.equal(edge_attr1, edge_attr2):
            return False

    return True


@overload
def to_undirected(
    edge_index: Tensor,
    edge_attr: str = MISSING,
    num_nodes: Optional[int] = None,
    reduce: str = 'add',
) -> Tensor:
    pass


@overload
def to_undirected(  # noqa: F811
    edge_index: Tensor,
    edge_attr: Tensor,
    num_nodes: Optional[int] = None,
    reduce: str = 'add',
) -> Tuple[Tensor, Tensor]:
    pass


@overload
def to_undirected(  # noqa: F811
    edge_index: Tensor,
    edge_attr: Optional[Tensor],
    num_nodes: Optional[int] = None,
    reduce: str = 'add',
) -> Tuple[Tensor, Optional[Tensor]]:
    pass


@overload
def to_undirected(  # noqa: F811
    edge_index: Tensor,
    edge_attr: List[Tensor],
    num_nodes: Optional[int] = None,
    reduce: str = 'add',
) -> Tuple[Tensor, List[Tensor]]:
    pass


def to_undirected(  # noqa: F811
    edge_index: Tensor,
    edge_attr: Union[Optional[Tensor], List[Tensor], str] = MISSING,
    num_nodes: Optional[int] = None,
    reduce: str = 'add',
) -> Union[Tensor, Tuple[Tensor, OptTensor], Tuple[Tensor, List[Tensor]]]:
    r"""Converts the graph given by :attr:`edge_index` to an undirected graph
    such that :math:`(j,i) \in \mathcal{E}` for every edge :math:`(i,j) \in
    \mathcal{E}`.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor or List[Tensor], optional): Edge weights or multi-
            dimensional edge features.
            If given as a list, will remove duplicates for all its entries.
            (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max(edge_index) + 1`. (default: :obj:`None`)
        reduce (str, optional): The reduce operation to use for merging edge
            features (:obj:`"add"`, :obj:`"mean"`, :obj:`"min"`, :obj:`"max"`,
            :obj:`"mul"`). (default: :obj:`"add"`)

    :rtype: :class:`LongTensor` if :attr:`edge_attr` is not passed, else
        (:class:`LongTensor`, :obj:`Optional[Tensor]` or :obj:`List[Tensor]]`)

    .. warning::

        From :pyg:`PyG >= 2.3.0` onwards, this function will always return a
        tuple whenever :obj:`edge_attr` is passed as an argument (even in case
        it is set to :obj:`None`).

    Examples:
        >>> edge_index = torch.tensor([[0, 1, 1],
        ...                            [1, 0, 2]])
        >>> to_undirected(edge_index)
        tensor([[0, 1, 1, 2],
                [1, 0, 2, 1]])

        >>> edge_index = torch.tensor([[0, 1, 1],
        ...                            [1, 0, 2]])
        >>> edge_weight = torch.tensor([1., 1., 1.])
        >>> to_undirected(edge_index, edge_weight)
        (tensor([[0, 1, 1, 2],
                [1, 0, 2, 1]]),
        tensor([2., 2., 1., 1.]))

        >>> # Use 'mean' operation to merge edge features
        >>>  to_undirected(edge_index, edge_weight, reduce='mean')
        (tensor([[0, 1, 1, 2],
                [1, 0, 2, 1]]),
        tensor([1., 1., 1., 1.]))
    """
    # Maintain backward compatibility to `to_undirected(edge_index, num_nodes)`
    if isinstance(edge_attr, int):
        num_nodes = edge_attr
        edge_attr = MISSING

    row, col = edge_index[0], edge_index[1]
    row, col = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)
    edge_index = torch.stack([row, col], dim=0)

    if isinstance(edge_attr, Tensor):
        edge_attr = torch.cat([edge_attr, edge_attr], dim=0)
    elif isinstance(edge_attr, (list, tuple)):
        edge_attr = [torch.cat([e, e], dim=0) for e in edge_attr]

    return coalesce(edge_index, edge_attr, num_nodes, reduce)
