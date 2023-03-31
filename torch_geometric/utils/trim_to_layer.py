from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch_sparse import SparseStorage, SparseTensor

from torch_geometric.typing import (
    EdgeType,
    MaybeHeteroEdgeTensor,
    MaybeHeteroNodeTensor,
    NodeType,
)


def resize_adj_t(src: SparseTensor, new_num_rows: int,
                 active_nodes_num: None) -> SparseTensor:
    r"""It resizes a bidimensional :obj:`src` SparseTensor
    along both dimensions and it returns a squared SparseTensor.
    Resizing always starts from position 0,0 as it is
    assumed that src is an adj_t matrix obtained via
    BFS traversing of a graph batch, BFS starting from
    the nodes that have been initially selected (target nodes)
    for the graph batch formation
    Args:
        src (SparseTensor): SparseTensor to be manipulated
        new_num_rows (int): last position to include in the output
        active_nodes_num (int): number of nodes we want to compute
        representation for
    """
    if len(src.storage._sparse_sizes) != 2:
        raise NotImplementedError

    start = 0
    if src.storage._rowptr is None:
        rowptr, col, value = src.csr()

    # rowptr
    rowptr = torch.narrow(src.storage._rowptr, 0, start,
                          new_num_rows + 1).clone()
    rowptr[(active_nodes_num + 1):] = rowptr[active_nodes_num]

    # col and value
    col = torch.narrow(src.storage._col, 0, start, rowptr[-1])
    value = torch.narrow(
        src.storage._value,
        0,
        start,
        rowptr[-1])\
        if src.storage._value is not None else None

    # indices for conversion to csc
    csr2csc = src.storage._csr2csc[src.storage._csr2csc < len(col)] \
        if src.storage._csr2csc is not None else None

    # update storage and edge_index
    storage = SparseStorage(row=None, rowptr=rowptr, col=col, value=value,
                            sparse_sizes=(new_num_rows, new_num_rows),
                            rowcount=None, colptr=None, colcount=None,
                            csr2csc=csr2csc, csc2csr=None, is_sorted=True,
                            trust_data=False)
    return src.from_storage(storage)


def trim_to_layer(
    layer: int,
    num_sampled_nodes_per_hop: Union[List[int], Dict[NodeType, List[int]]],
    num_sampled_edges_per_hop: Union[List[int], Dict[EdgeType, List[int]]],
    x: Union[MaybeHeteroNodeTensor],
    edge_index: Union[MaybeHeteroEdgeTensor],
    edge_attr: Optional[MaybeHeteroEdgeTensor] = None,
) -> Tuple[MaybeHeteroEdgeTensor, MaybeHeteroNodeTensor,
           Optional[MaybeHeteroEdgeTensor]]:
    r"""Trims the :obj:`edge_index` representation, node features :obj:`x` and
    edge features :obj:`edge_attr` to a minimal-sized representation for the
    current GNN layer :obj:`layer` in directed
    :class:`~torch_geometric.loader.NeighborLoader` scenarios.

    This ensures that no computation is performed for nodes and edges that are
    not included in the current GNN layer, thus avoiding unnecessary
    computation within the GNN when performing neighborhood sampling.

    Args:
        layer (int): The current GNN layer.
        num_sampled_nodes_per_hop (List[int] or Dict[NodeType, List[int]]): The
            number of sampled nodes per hop.
        num_sampled_edges_per_hop (List[int] or Dict[EdgeType, List[int]]): The
            number of sampled edges per hop.
        x (torch.Tensor or Dict[NodeType, torch.Tensor]): The homogeneous or
            heterogeneous (hidden) node features.
        edge_index (torch.Tensor or Dict[EdgeType, torch.Tensor]): The
            homogeneous or heterogeneous edge indices.
        edge_attr (torch.Tensor or Dict[EdgeType, torch.Tensor], optional): The
            homogeneous or heterogeneous (hidden) edge features.
    """
    if layer <= 0:
        return x, edge_index, edge_attr

    if isinstance(num_sampled_edges_per_hop, dict):
        x = {
            k: v.narrow(
                dim=0,
                start=0,
                length=v.size(0) - num_sampled_nodes_per_hop[k][-layer],
            )
            for k, v in x.items()
        }
        edge_index = {
            k: v.narrow(
                dim=1,
                start=0,
                length=v.size(1) - num_sampled_edges_per_hop[k][-layer],
            )
            for k, v in edge_index.items()
        }
        if edge_attr is not None:
            edge_attr = {
                k: v.narrow(
                    dim=0,
                    start=0,
                    length=v.size(0) - num_sampled_edges_per_hop[k][-layer],
                )
                for k, v in edge_attr.items()
            }
        return x, edge_index, edge_attr

    # this point should be reached by homogeneous case only,
    # hetero is handled above
    # for homogeneous case, support for the SparseTensor case
    # (from torch_sparse) is provided
    # whether or not edge_index is Tensor or SparseTensor
    # x and edge_attr should be treated in same way
    x = x.narrow(
        dim=0,
        start=0,
        length=x.size(0) - num_sampled_nodes_per_hop[-layer],
    )
    if edge_attr is not None:
        edge_attr = edge_attr.narrow(
            dim=0,
            start=0,
            length=edge_attr.size(0) - num_sampled_edges_per_hop[-layer],
        )
    # adj matrix as Tensor
    if isinstance(edge_index, Tensor):
        edge_index = edge_index.narrow(
            dim=1,
            start=0,
            length=edge_index.size(1) - num_sampled_edges_per_hop[-layer],
        )
        return x, edge_index, edge_attr

    # adj matrix as SparseTensor
    if isinstance(edge_index, SparseTensor):
        if edge_index.storage._rowptr is None:
            edge_index.storage._rowptr,\
                edge_index.storage._col,\
                edge_index.storage._value = edge_index.csr()

        new_num_rows = edge_index.storage._sparse_sizes[0] \
            - num_sampled_nodes_per_hop[-layer]

        active_nodes_num = new_num_rows - \
            num_sampled_nodes_per_hop[-(layer + 1)]

        edge_index = resize_adj_t(edge_index, new_num_rows=new_num_rows,
                                  active_nodes_num=active_nodes_num)
        return x, edge_index, edge_attr
    raise NotImplementedError  # end of trim_to_layer


class TrimToLayer(torch.nn.Module):
    def forward(
        self,
        layer: int,
        num_sampled_nodes_per_hop: Optional[List[int]],
        num_sampled_edges_per_hop: Optional[List[int]],
        x: Tensor,
        edge_index: Union[Tensor, SparseTensor],
        edge_attr: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:

        if (not isinstance(num_sampled_nodes_per_hop, list)
                and isinstance(num_sampled_edges_per_hop, list)):
            raise ValueError("'num_sampled_nodes_per_hop' needs to be given")
        if (not isinstance(num_sampled_edges_per_hop, list)
                and isinstance(num_sampled_nodes_per_hop, list)):
            raise ValueError("'num_sampled_edges_per_hop' needs to be given")

        if num_sampled_nodes_per_hop is None:
            return x, edge_index, edge_attr
        if num_sampled_edges_per_hop is None:
            return x, edge_index, edge_attr

        return trim_to_layer(
            layer,
            num_sampled_nodes_per_hop,
            num_sampled_edges_per_hop,
            x,
            edge_index,
            edge_attr,
        )
