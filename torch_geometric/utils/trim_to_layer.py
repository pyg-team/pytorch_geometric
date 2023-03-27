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
    
    # this point should be reached by homogeneous case only, hetero is handled above
    # for homogeneous case, I add here support for the SparseTensor (from torch_sparse) case as well  
    # x and edge_attr should be treated in same way, regardles we use SparseTensor or Tensor to represent adj matrix
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
            A = edge_index.csr()
            (edge_index.storage._rowptr, 
            edge_index.storage._col, 
            edge_index.storage._value) = A

        new_num_rows = edge_index.storage._sparse_sizes[0] - \
                num_sampled_nodes_per_hop[-layer]
        sparse_sizes = (new_num_rows, new_num_rows)  # n_rows == n_cols

        # rowptr
        rowptr = torch.narrow(edge_index.storage._rowptr, 0, 0,
                                new_num_rows + 1).clone()
        active_nodes_num = new_num_rows - num_sampled_nodes_per_hop[-(layer + 1)]
        rowptr[(active_nodes_num + 1):] = rowptr[active_nodes_num]

        # col and value
        col = torch.narrow(edge_index.storage._col, 0, 0, rowptr[-1])
        value = torch.narrow(edge_index.storage._value, 0, 0, rowptr[-1]
                                ) if edge_index.storage._value is not None else None

        # indeces for conversion to csc
        csr2csc = edge_index.storage._csr2csc[edge_index.storage._csr2csc < len(col)] \
            if edge_index.storage._csr2csc is not None else None

        # update storage and edge_index
        storage = SparseStorage(row=None, rowptr=rowptr, col=col,
                                value=value, sparse_sizes=sparse_sizes,
                                rowcount=None, colptr=None,
                                colcount=None, csr2csc=csr2csc,
                                csc2csr=None, is_sorted=True,
                                trust_data=False)
        edge_index = SparseTensor.from_storage(storage)
        return x, edge_index, edge_attr

    raise NotImplemented
    # end of trim_to_layer

class TrimToLayer(torch.nn.Module):
    def forward(
        self,
        layer: int,
        num_sampled_nodes_per_hop: Optional[List[int]],
        num_sampled_edges_per_hop: Optional[List[int]],
        x: Tensor,
        edge_index: Tensor,
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
