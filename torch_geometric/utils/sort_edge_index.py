from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor

from torch_geometric.typing import OptTensor, EdgeType, SparseTensor
from torch_geometric.utils import index_sort
from torch_geometric.utils.num_nodes import maybe_num_nodes

MISSING = '???'


@torch.jit._overload
def sort_edge_index(edge_index, edge_attr, num_nodes, sort_by_row):
    # type: (Tensor, str, Optional[int], bool) -> Tensor  # noqa
    pass


@torch.jit._overload
def sort_edge_index(edge_index, edge_attr, num_nodes, sort_by_row):
    # type: (Tensor, Optional[Tensor], Optional[int], bool) -> Tuple[Tensor, Optional[Tensor]]  # noqa
    pass


@torch.jit._overload
def sort_edge_index(edge_index, edge_attr, num_nodes, sort_by_row):
    # type: (Tensor, List[Tensor], Optional[int], bool) -> Tuple[Tensor, List[Tensor]]  # noqa
    pass


def sort_edge_index(
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
            :obj:`edge_index` column-wise.

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


def construct_edge_index(
    self,
    edge_index_dict: Dict[EdgeType, Adj],
    src_offset: Dict[EdgeType, int],
    dst_offset: [NodeType, int],
    edge_attr_dict: Dict[Union[EdgeType, str], Tensor] = None,
) -> Tuple[Adj, OptTensor]:
    """Constructs a tensor of edge indices by concatenating edge indices
    for each edge type. The edge indices are increased by the offset of the
    source and destination nodes."""
    edge_indices: List[Tensor] = []
    if edge_attr_dict is not None:
        edge_attrs: List[Tensor] = []
        use_e_attrs = True
    else:
        use_e_attrs = False

    for edge_type in self.edge_types:
        _, _, dst_type = edge_type

        edge_index = edge_index_dict[edge_type]

        # (TODO) Add support for SparseTensor w/o converting.
        is_sparse = isinstance(edge_index, SparseTensor)
        if is_sparse:  # Convert to COO
            dst, src, _ = edge_index.coo()
            edge_index = torch.stack([src, dst], dim=0)
        else:
            edge_index = edge_index.clone()
        if use_e_attrs:
            if isinstance(list(edge_attr_dict.keys())[0], EdgeType):
                edge_attr_i = edge_attr_dict[edge_type].expand(
                    edge_index.size(1), -1)
            else:
                # Param dicts dont like tuple keys
                edge_attr_i = edge_attr_dict['__'.join(edge_type)].expand(
                    edge_index.size(1), -1)

            edge_attrs.append(edge_attr_i)

        # Add offset to edge indices:
        edge_index[0] += src_offset[edge_type]
        edge_index[1] += dst_offset[dst_type]
        edge_indices.append(edge_index)

    # Concatenate all edges and edge tensors:
    if use_e_attrs:
        edge_attr = torch.cat(edge_attrs, dim=0)
    else:
        edge_attr = None
    edge_index = torch.cat(edge_indices, dim=1)

    if is_sparse:
        edge_index = SparseTensor(row=edge_index[1], col=edge_index[0],
                                  value=edge_attr if use_e_attrs else None)

    return edge_index, edge_attr
