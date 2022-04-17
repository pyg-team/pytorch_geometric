from typing import List, Optional, Tuple, Union

from torch import Tensor

from .num_nodes import maybe_num_nodes


def sort_edge_index(
    edge_index: Tensor,
    edge_attr: Optional[Union[Tensor, List[Tensor]]] = None,
    num_nodes: Optional[int] = None,
    sort_by_row: bool = True,
) -> Union[Tensor, Tuple[Tensor, Tensor], Tuple[Tensor, List[Tensor]]]:
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
            :obj:`edge_index` column-wise.

    :rtype: :class:`LongTensor` if :attr:`edge_attr` is :obj:`None`, else
        (:class:`LongTensor`, :obj:`Tensor` or :obj:`List[Tensor]]`)
    """
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    idx = edge_index[1 - int(sort_by_row)] * num_nodes
    idx += edge_index[int(sort_by_row)]

    perm = idx.argsort()

    edge_index = edge_index[:, perm]

    if edge_attr is None:
        return edge_index
    elif isinstance(edge_attr, Tensor):
        return edge_index, edge_attr[perm]
    else:
        return edge_index, [e[perm] for e in edge_attr]
