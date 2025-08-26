import typing
from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor

import torch_geometric.typing
from torch_geometric import EdgeIndex
from torch_geometric.edge_index import SortOrder
from torch_geometric.typing import OptTensor
from torch_geometric.utils import index_sort, lexsort
from torch_geometric.utils.num_nodes import maybe_num_nodes

if typing.TYPE_CHECKING:
    from typing import overload
else:
    from torch.jit import _overload as overload

MISSING = '???'


@overload
def sort_edge_index(
    edge_index: Tensor,
    edge_attr: str = MISSING,
    num_nodes: Optional[int] = None,
    sort_by_row: bool = True,
) -> Tensor:
    pass


@overload
def sort_edge_index(  # noqa: F811
    edge_index: Tensor,
    edge_attr: Tensor,
    num_nodes: Optional[int] = None,
    sort_by_row: bool = True,
) -> Tuple[Tensor, Tensor]:
    pass


@overload
def sort_edge_index(  # noqa: F811
    edge_index: Tensor,
    edge_attr: OptTensor,
    num_nodes: Optional[int] = None,
    sort_by_row: bool = True,
) -> Tuple[Tensor, OptTensor]:
    pass


@overload
def sort_edge_index(  # noqa: F811
    edge_index: Tensor,
    edge_attr: List[Tensor],
    num_nodes: Optional[int] = None,
    sort_by_row: bool = True,
) -> Tuple[Tensor, List[Tensor]]:
    pass


def sort_edge_index(  # noqa: F811
    edge_index: Tensor,
    edge_attr: Union[OptTensor, List[Tensor], str] = MISSING,
    num_nodes: Optional[int] = None,
    sort_by_row: bool = True,
) -> Union[Tensor, Tuple[Tensor, OptTensor], Tuple[Tensor, List[Tensor]]]:
    """Row-wise sorts :obj:`edge_index`.

    Args:
        edge_index (torch.Tensor): The edge indices.
        edge_attr (torch.Tensor or List[torch.Tensor], optional): Edge weights
            or multi-dimensional edge features.
            If given as a list, will re-shuffle and remove duplicates for all
            its entries. (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        sort_by_row (bool, optional): If set to :obj:`False`, will sort
            :obj:`edge_index` column-wise/by destination node.
            (default: :obj:`True`)

    :rtype: :class:`LongTensor` if :attr:`edge_attr` is not passed, else
        (:class:`LongTensor`, :obj:`Optional[Tensor]` or :obj:`List[Tensor]]`)

    .. warning::

        From :pyg:`PyG >= 2.3.0` onwards, this function will always return a
        tuple whenever :obj:`edge_attr` is passed as an argument (even in case
        it is set to :obj:`None`).

    Examples:
        >>> edge_index = torch.tensor([[2, 1, 1, 0],
                                [1, 2, 0, 1]])
        >>> edge_attr = torch.tensor([[1], [2], [3], [4]])
        >>> sort_edge_index(edge_index)
        tensor([[0, 1, 1, 2],
                [1, 0, 2, 1]])

        >>> sort_edge_index(edge_index, edge_attr)
        (tensor([[0, 1, 1, 2],
                [1, 0, 2, 1]]),
        tensor([[4],
                [3],
                [2],
                [1]]))
    """
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    if num_nodes * num_nodes > torch_geometric.typing.MAX_INT64:
        perm = lexsort(keys=[
            edge_index[int(sort_by_row)],
            edge_index[1 - int(sort_by_row)],
        ])
    else:
        idx = edge_index[1 - int(sort_by_row)] * num_nodes
        idx += edge_index[int(sort_by_row)]
        _, perm = index_sort(idx, max_value=num_nodes * num_nodes)

    if isinstance(edge_index, Tensor):
        is_undirected = False
        if not torch.jit.is_scripting() and isinstance(edge_index, EdgeIndex):
            is_undirected = edge_index.is_undirected
        edge_index = edge_index[:, perm]
        if not torch.jit.is_scripting() and isinstance(edge_index, EdgeIndex):
            edge_index._sort_order = SortOrder('row' if sort_by_row else 'col')
            edge_index._is_undirected = is_undirected
    elif isinstance(edge_index, tuple):
        edge_index = (edge_index[0][perm], edge_index[1][perm])
    else:
        raise NotImplementedError

    if edge_attr is None:
        return edge_index, None
    if isinstance(edge_attr, Tensor):
        return edge_index, edge_attr[perm]
    if isinstance(edge_attr, (list, tuple)):
        return edge_index, [e[perm] for e in edge_attr]

    return edge_index
