from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor

from torch_geometric.utils import scatter

from .num_nodes import maybe_num_nodes


@torch.jit._overload
def coalesce(edge_index, edge_attr=None, num_nodes=None, reduce="add",
             is_sorted=False, sort_by_row=True):
    # type: (Tensor, Optional[bool], Optional[int], str, bool, bool) -> Tensor  # noqa
    pass


@torch.jit._overload
def coalesce(edge_index, edge_attr=None, num_nodes=None, reduce="add",
             is_sorted=False, sort_by_row=True):
    # type: (Tensor, Tensor, Optional[int], str, bool, bool) -> Tuple[Tensor, Tensor]  # noqa
    pass


@torch.jit._overload
def coalesce(edge_index, edge_attr=None, num_nodes=None, reduce="add",
             is_sorted=False, sort_by_row=True):
    # type: (Tensor, List[Tensor], Optional[int], str, bool, bool) -> Tuple[Tensor, List[Tensor]]  # noqa
    pass


def coalesce(
    edge_index: Tensor,
    edge_attr: Union[Optional[Tensor], List[Tensor]] = None,
    num_nodes: Optional[int] = None,
    reduce: str = "add",
    is_sorted: bool = False,
    sort_by_row: bool = True,
) -> Union[Tensor, Tuple[Tensor, Tensor], Tuple[Tensor, List[Tensor]]]:
    """Row-wise sorts :obj:`edge_index` and removes its duplicated entries.
    Duplicate entries in :obj:`edge_attr` are merged by scattering them
    together according to the given :obj:`reduce` option.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor or List[Tensor], optional): Edge weights or multi-
            dimensional edge features.
            If given as a list, will re-shuffle and remove duplicates for all
            its entries. (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        reduce (string, optional): The reduce operation to use for merging edge
            features (:obj:`"add"`, :obj:`"mean"`, :obj:`"min"`, :obj:`"max"`,
            :obj:`"mul"`). (default: :obj:`"add"`)
        is_sorted (bool, optional): If set to :obj:`True`, will expect
            :obj:`edge_index` to be already sorted row-wise.
        sort_by_row (bool, optional): If set to :obj:`False`, will sort
            :obj:`edge_index` column-wise.

    :rtype: :class:`LongTensor` if :attr:`edge_attr` is :obj:`None`, else
        (:class:`LongTensor`, :obj:`Tensor` or :obj:`List[Tensor]]`)

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
    nnz = edge_index.size(1)
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    idx = edge_index.new_empty(nnz + 1)
    idx[0] = -1
    idx[1:] = edge_index[1 - int(sort_by_row)]
    idx[1:].mul_(num_nodes).add_(edge_index[int(sort_by_row)])

    if not is_sorted:
        idx[1:], perm = idx[1:].sort()
        edge_index = edge_index[:, perm]
        if isinstance(edge_attr, Tensor):
            edge_attr = edge_attr[perm]
        elif isinstance(edge_attr, (list, tuple)):
            edge_attr = [e[perm] for e in edge_attr]

    mask = idx[1:] > idx[:-1]

    # Only perform expensive merging in case there exists duplicates:
    if mask.all():
        if isinstance(edge_attr, (Tensor, list, tuple)):
            return edge_index, edge_attr
        return edge_index

    edge_index = edge_index[:, mask]

    if edge_attr is None:
        return edge_index

    dim_size = edge_index.size(1)
    idx = torch.arange(0, nnz, device=edge_index.device)
    idx.sub_(mask.logical_not_().cumsum(dim=0))

    if isinstance(edge_attr, Tensor):
        edge_attr = scatter(edge_attr, idx, 0, dim_size, reduce)
        return edge_index, edge_attr
    elif isinstance(edge_attr, (list, tuple)):
        edge_attr = [scatter(e, idx, 0, dim_size, reduce) for e in edge_attr]
        return edge_index, edge_attr

    return edge_index
