from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor

from torch_geometric.typing import (
    Adj,
    EdgeType,
    MaybeHeteroEdgeTensor,
    MaybeHeteroNodeTensor,
    NodeType,
    SparseStorage,
    SparseTensor,
)


def trim_to_layer(
    layer: int,
    num_sampled_nodes_per_hop: Union[List[int], Dict[NodeType, List[int]]],
    num_sampled_edges_per_hop: Union[List[int], Dict[EdgeType, List[int]]],
    x: MaybeHeteroNodeTensor,
    edge_index: MaybeHeteroEdgeTensor,
    edge_attr: Optional[MaybeHeteroEdgeTensor] = None,
) -> Tuple[MaybeHeteroNodeTensor, MaybeHeteroEdgeTensor,
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
            k: trim_feat(v, layer, num_sampled_nodes_per_hop[k])
            for k, v in x.items()
        }
        edge_index = {
            k:
            trim_adj(
                v,
                layer,
                num_sampled_nodes_per_hop[k[0]],
                num_sampled_nodes_per_hop[k[-1]],
                num_sampled_edges_per_hop[k],
            )
            for k, v in edge_index.items()
        }
        if edge_attr is not None:
            edge_attr = {
                k: trim_feat(v, layer, num_sampled_edges_per_hop[k])
                for k, v in edge_attr.items()
            }
        return x, edge_index, edge_attr

    x = trim_feat(x, layer, num_sampled_nodes_per_hop)
    edge_index = trim_adj(
        edge_index,
        layer,
        num_sampled_nodes_per_hop,
        num_sampled_nodes_per_hop,
        num_sampled_edges_per_hop,
    )

    if edge_attr is not None:
        edge_attr = trim_feat(edge_attr, layer, num_sampled_edges_per_hop)

    return x, edge_index, edge_attr


class TrimToLayer(torch.nn.Module):
    def forward(
        self,
        layer: int,
        num_sampled_nodes_per_hop: Optional[List[int]],
        num_sampled_edges_per_hop: Optional[List[int]],
        x: Tensor,
        edge_index: Adj,
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


# Helper functions ############################################################


def trim_feat(x: Tensor, layer: int, num_samples_per_hop: List[int]) -> Tensor:
    if layer <= 0:
        return x

    return x.narrow(
        dim=0,
        start=0,
        length=x.size(0) - num_samples_per_hop[-layer],
    )


def trim_adj(
    edge_index: Adj,
    layer: int,
    num_sampled_src_nodes_per_hop: List[int],
    num_sampled_dst_nodes_per_hop: List[int],
    num_sampled_edges_per_hop: List[int],
) -> Adj:

    if layer <= 0:
        return edge_index

    if isinstance(edge_index, Tensor):
        return edge_index.narrow(
            dim=1,
            start=0,
            length=edge_index.size(1) - num_sampled_edges_per_hop[-layer],
        )

    elif isinstance(edge_index, SparseTensor):
        size = (
            edge_index.size(0) - num_sampled_dst_nodes_per_hop[-layer],
            edge_index.size(1) - num_sampled_src_nodes_per_hop[-layer],
        )

        num_seed_nodes = size[0] - num_sampled_dst_nodes_per_hop[-(layer + 1)]

        return trim_sparse_tensor(edge_index, size, num_seed_nodes)

    raise ValueError(f"Unsupported 'edge_index' type '{type(edge_index)}'")


def trim_sparse_tensor(src: SparseTensor, size: Tuple[int, int],
                       num_seed_nodes: int) -> SparseTensor:
    r"""Trims a :class:`SparseTensor` along both dimensions to only contain
    the upper :obj:`num_nodes` in both dimensions.

    It is assumed that :class:`SparseTensor` is obtained from BFS traversing,
    starting from the nodes that have been initially selected.

    Args:
        src (SparseTensor): The sparse tensor.
        size (Tuple[int, int]): The number of source and destination nodes to
            keep.
        num_seed_nodes (int): The number of seed nodes to compute
            representations.
    """
    rowptr, col, value = src.csr()

    rowptr = torch.narrow(rowptr, 0, 0, size[0] + 1).clone()
    rowptr[num_seed_nodes + 1:] = rowptr[num_seed_nodes]

    col = torch.narrow(col, 0, 0, rowptr[-1])

    if value is not None:
        value = torch.narrow(value, 0, 0, rowptr[-1])

    csr2csc = src.storage._csr2csc
    if csr2csc is not None:
        csr2csc = csr2csc[csr2csc < len(col)]

    storage = SparseStorage(
        row=None,
        rowptr=rowptr,
        col=col,
        value=value,
        sparse_sizes=size,
        rowcount=None,
        colptr=None,
        colcount=None,
        csr2csc=csr2csc,
        csc2csr=None,
        is_sorted=True,
        trust_data=True,
    )
    return src.from_storage(storage)
