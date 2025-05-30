from typing import Dict, List, Optional, Tuple, Union, overload

import torch
from torch import Tensor

from torch_geometric import EdgeIndex
from torch_geometric.typing import (
    Adj,
    EdgeType,
    MaybeHeteroAdjTensor,
    MaybeHeteroEdgeTensor,
    MaybeHeteroNodeTensor,
    NodeType,
    SparseStorage,
    SparseTensor,
)


@overload
def trim_to_layer(
    layer: int,
    num_sampled_nodes_per_hop: List[int],
    num_sampled_edges_per_hop: List[int],
    x: Tensor,
    edge_index: Adj,
    edge_attr: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    pass


@overload
def trim_to_layer(
    layer: int,
    num_sampled_nodes_per_hop: Dict[NodeType, List[int]],
    num_sampled_edges_per_hop: Dict[EdgeType, List[int]],
    x: Dict[NodeType, Tensor],
    edge_index: Dict[EdgeType, Adj],
    edge_attr: Optional[Dict[EdgeType, Tensor]] = None,
) -> Tuple[Dict[NodeType, Tensor], Dict[EdgeType, Adj], Optional[Dict[
        EdgeType, Tensor]]]:
    pass


def trim_to_layer(
    layer: int,
    num_sampled_nodes_per_hop: Union[List[int], Dict[NodeType, List[int]]],
    num_sampled_edges_per_hop: Union[List[int], Dict[EdgeType, List[int]]],
    x: MaybeHeteroNodeTensor,
    edge_index: MaybeHeteroEdgeTensor,
    edge_attr: Optional[MaybeHeteroEdgeTensor] = None,
) -> Tuple[MaybeHeteroNodeTensor, MaybeHeteroAdjTensor,
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
        assert isinstance(num_sampled_nodes_per_hop, dict)

        assert isinstance(x, dict)
        x = {
            k: trim_feat(v, layer, num_sampled_nodes_per_hop[k])
            for k, v in x.items()
        }

        assert isinstance(edge_index, dict)
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
            assert isinstance(edge_attr, dict)
            edge_attr = {
                k: trim_feat(v, layer, num_sampled_edges_per_hop[k])
                for k, v in edge_attr.items()
            }

        return x, edge_index, edge_attr

    assert isinstance(num_sampled_nodes_per_hop, list)

    assert isinstance(x, Tensor)
    x = trim_feat(x, layer, num_sampled_nodes_per_hop)

    assert isinstance(edge_index, (Tensor, SparseTensor))
    edge_index = trim_adj(
        edge_index,
        layer,
        num_sampled_nodes_per_hop,
        num_sampled_nodes_per_hop,
        num_sampled_edges_per_hop,
    )

    if edge_attr is not None:
        assert isinstance(edge_attr, Tensor)
        edge_attr = trim_feat(edge_attr, layer, num_sampled_edges_per_hop)

    return x, edge_index, edge_attr


class TrimToLayer(torch.nn.Module):
    @torch.jit.unused
    def forward(
        self,
        layer: int,
        num_sampled_nodes_per_hop: Optional[List[int]],
        num_sampled_edges_per_hop: Optional[List[int]],
        x: Tensor,
        edge_index: Adj,
        edge_attr: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Adj, Optional[Tensor]]:

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
        edge_index = edge_index.narrow(
            dim=1,
            start=0,
            length=edge_index.size(1) - num_sampled_edges_per_hop[-layer],
        )
        if not torch.jit.is_scripting() and isinstance(edge_index, EdgeIndex):
            num_rows, num_cols = edge_index.sparse_size()
            if num_rows is not None:
                num_rows -= num_sampled_src_nodes_per_hop[-layer]
            if num_cols is not None:
                num_cols -= num_sampled_dst_nodes_per_hop[-layer]
            edge_index.sparse_resize_(num_rows, num_cols)
        return edge_index

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

    col = torch.narrow(col, 0, 0, rowptr[-1])  # type: ignore

    if value is not None:
        value = torch.narrow(value, 0, 0, rowptr[-1])  # type: ignore

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
