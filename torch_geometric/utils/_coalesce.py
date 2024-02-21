import typing
from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor

import torch_geometric.typing
from torch_geometric import EdgeIndex
from torch_geometric.edge_index import SortOrder
from torch_geometric.typing import OptTensor
from torch_geometric.utils import index_sort, scatter
from torch_geometric.utils.num_nodes import maybe_num_nodes

if typing.TYPE_CHECKING:
    from typing import overload
else:
    from torch.jit import _overload as overload

MISSING = '???'


@overload
def coalesce(
    edge_index: Tensor,
    edge_attr: str = MISSING,
    num_nodes: Optional[int] = None,
    reduce: str = 'sum',
    is_sorted: bool = False,
    sort_by_row: bool = True,
) -> Tensor:
    pass


@overload
def coalesce(  # noqa: F811
    edge_index: Tensor,
    edge_attr: Tensor,
    num_nodes: Optional[int] = None,
    reduce: str = 'sum',
    is_sorted: bool = False,
    sort_by_row: bool = True,
) -> Tuple[Tensor, Tensor]:
    pass


@overload
def coalesce(  # noqa: F811
    edge_index: Tensor,
    edge_attr: OptTensor,
    num_nodes: Optional[int] = None,
    reduce: str = 'sum',
    is_sorted: bool = False,
    sort_by_row: bool = True,
) -> Tuple[Tensor, OptTensor]:
    pass


@overload
def coalesce(  # noqa: F811
    edge_index: Tensor,
    edge_attr: List[Tensor],
    num_nodes: Optional[int] = None,
    reduce: str = 'sum',
    is_sorted: bool = False,
    sort_by_row: bool = True,
) -> Tuple[Tensor, List[Tensor]]:
    pass


def coalesce(  # noqa: F811
    edge_index: Tensor,
    edge_attr: Union[OptTensor, List[Tensor], str] = MISSING,
    num_nodes: Optional[int] = None,
    reduce: str = 'sum',
    is_sorted: bool = False,
    sort_by_row: bool = True,
) -> Union[Tensor, Tuple[Tensor, OptTensor], Tuple[Tensor, List[Tensor]]]:
    """Row-wise sorts :obj:`edge_index` and removes its duplicated entries.
    Duplicate entries in :obj:`edge_attr` are merged by scattering them
    together according to the given :obj:`reduce` option.

    Args:
        edge_index (torch.Tensor): The edge indices.
        edge_attr (torch.Tensor or List[torch.Tensor], optional): Edge weights
            or multi-dimensional edge features.
            If given as a list, will re-shuffle and remove duplicates for all
            its entries. (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        reduce (str, optional): The reduce operation to use for merging edge
            features (:obj:`"sum"`, :obj:`"mean"`, :obj:`"min"`, :obj:`"max"`,
            :obj:`"mul"`, :obj:`"any"`). (default: :obj:`"sum"`)
        is_sorted (bool, optional): If set to :obj:`True`, will expect
            :obj:`edge_index` to be already sorted row-wise.
        sort_by_row (bool, optional): If set to :obj:`False`, will sort
            :obj:`edge_index` column-wise.

    :rtype: :class:`LongTensor` if :attr:`edge_attr` is not passed, else
        (:class:`LongTensor`, :obj:`Optional[Tensor]` or :obj:`List[Tensor]]`)

    .. warning::

        From :pyg:`PyG >= 2.3.0` onwards, this function will always return a
        tuple whenever :obj:`edge_attr` is passed as an argument (even in case
        it is set to :obj:`None`).

    Example:
        >>> edge_index = torch.tensor([[1, 1, 2, 3],
        ...                            [3, 3, 1, 2]])
        >>> edge_attr = torch.tensor([1., 1., 1., 1.])
        >>> coalesce(edge_index)
        tensor([[1, 2, 3],
                [3, 1, 2]])

        >>> # Sort `edge_index` column-wise
        >>> coalesce(edge_index, sort_by_row=False)
        tensor([[2, 3, 1],
                [1, 2, 3]])

        >>> coalesce(edge_index, edge_attr)
        (tensor([[1, 2, 3],
                [3, 1, 2]]),
        tensor([2., 1., 1.]))

        >>> # Use 'mean' operation to merge edge features
        >>> coalesce(edge_index, edge_attr, reduce='mean')
        (tensor([[1, 2, 3],
                [3, 1, 2]]),
        tensor([1., 1., 1.]))
    """
    num_edges = edge_index[0].size(0)
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    if num_nodes * num_nodes > torch_geometric.typing.MAX_INT64:
        raise ValueError("'coalesce' will result in an overflow")

    idx = edge_index[0].new_empty(num_edges + 1)
    idx[0] = -1
    idx[1:] = edge_index[1 - int(sort_by_row)]
    idx[1:].mul_(num_nodes).add_(edge_index[int(sort_by_row)])

    is_undirected = False
    if not torch.jit.is_scripting() and isinstance(edge_index, EdgeIndex):
        is_undirected = edge_index.is_undirected

    if not is_sorted:
        idx[1:], perm = index_sort(idx[1:], max_value=num_nodes * num_nodes)
        if isinstance(edge_index, Tensor):
            edge_index = edge_index[:, perm]
        elif isinstance(edge_index, tuple):
            edge_index = (edge_index[0][perm], edge_index[1][perm])
        else:
            raise NotImplementedError
        if isinstance(edge_attr, Tensor):
            edge_attr = edge_attr[perm]
        elif isinstance(edge_attr, (list, tuple)):
            edge_attr = [e[perm] for e in edge_attr]

    if not torch.jit.is_scripting() and isinstance(edge_index, EdgeIndex):
        edge_index._sort_order = SortOrder('row' if sort_by_row else 'col')
        edge_index._is_undirected = is_undirected

    mask = idx[1:] > idx[:-1]

    # Only perform expensive merging in case there exists duplicates:
    if mask.all():
        if edge_attr is None or isinstance(edge_attr, Tensor):
            return edge_index, edge_attr
        if isinstance(edge_attr, (list, tuple)):
            return edge_index, edge_attr
        return edge_index

    if isinstance(edge_index, Tensor):
        edge_index = edge_index[:, mask]
        if not torch.jit.is_scripting() and isinstance(edge_index, EdgeIndex):
            edge_index._is_undirected = is_undirected
    elif isinstance(edge_index, tuple):
        edge_index = (edge_index[0][mask], edge_index[1][mask])
    else:
        raise NotImplementedError

    dim_size: Optional[int] = None
    if isinstance(edge_attr, (Tensor, list, tuple)) and len(edge_attr) > 0:
        dim_size = edge_index.size(1)
        idx = torch.arange(0, num_edges, device=edge_index.device)
        idx.sub_(mask.logical_not_().cumsum(dim=0))

    if edge_attr is None:
        return edge_index, None
    if isinstance(edge_attr, Tensor):
        edge_attr = scatter(edge_attr, idx, 0, dim_size, reduce)
        return edge_index, edge_attr
    if isinstance(edge_attr, (list, tuple)):
        if len(edge_attr) == 0:
            return edge_index, edge_attr
        edge_attr = [scatter(e, idx, 0, dim_size, reduce) for e in edge_attr]
        return edge_index, edge_attr

    return edge_index
