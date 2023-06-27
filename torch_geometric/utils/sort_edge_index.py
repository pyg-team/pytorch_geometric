from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor

from torch_geometric.typing import OptTensor
from torch_geometric.utils import index_sort
from torch_geometric.utils.num_nodes import maybe_num_nodes

MISSING = '???'


@torch.jit._overload
def sort_edge_index(edge_index, edge_attr, num_nodes, sort_by_row):  # noqa
    # type: (Tensor, str, Optional[int], bool) -> Tensor  # noqa
    pass


@torch.jit._overload
def sort_edge_index(edge_index, edge_attr, num_nodes, sort_by_row):  # noqa
    # type: (Tensor, Optional[Tensor], Optional[int], bool) -> Tuple[Tensor, Optional[Tensor]]  # noqa
    pass


@torch.jit._overload
def sort_edge_index(edge_index, edge_attr, num_nodes, sort_by_row):  # noqa
    # type: (Tensor, List[Tensor], Optional[int], bool) -> Tuple[Tensor, List[Tensor]]  # noqa
    pass


def sort_edge_index(  # noqa
    edge_index: Tensor,
    edge_attr: Union[OptTensor, List[Tensor], str] = MISSING,
    num_nodes: Optional[int] = None,
    sort_by_row: bool = True,
) -> Union[Tensor, Tuple[Tensor, OptTensor], Tuple[Tensor, List[Tensor]]]:
    """Row-wise sorts :obj:`edge_index`.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor or List[Tensor], optional): Edge weights or multi-
            dimensional edge features.
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

    idx = edge_index[1 - int(sort_by_row)] * num_nodes
    idx += edge_index[int(sort_by_row)]

    _, perm = index_sort(idx, max_value=num_nodes * num_nodes)

    edge_index = edge_index[:, perm]

    if edge_attr is None:
        return edge_index, None
    if isinstance(edge_attr, Tensor):
        return edge_index, edge_attr[perm]
    if isinstance(edge_attr, (list, tuple)):
        return edge_index, [e[perm] for e in edge_attr]

    return edge_index
