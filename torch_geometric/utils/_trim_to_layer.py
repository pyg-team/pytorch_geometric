import copy
from typing import Any, Dict, List, Optional, Tuple, Union, overload

import torch
from torch import Tensor

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


def filter_empty_entries(
        input_dict: Dict[Union[Any], Tensor]) -> Dict[Any, Tensor]:
    r"""Removes empty tensors from a dictionary. This avoids unnecessary
    computation when some node/edge types are non-reachable after trimming.
    """
    out_dict = copy.copy(input_dict)
    for key, value in input_dict.items():
        if value.numel() == 0:
            del out_dict[key]
    return out_dict


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
    if isinstance(num_sampled_edges_per_hop, dict):
        assert isinstance(num_sampled_nodes_per_hop, dict)

        assert isinstance(x, dict)
        # x trimming should be now implicit in message passing with rect adj mat
        # x = {
        #     k: trim_feat(v, layer, num_sampled_nodes_per_hop[k])
        #     for k, v in x.items()
        # }
        # x = filter_empty_entries(x)

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
        edge_index = filter_empty_entries(edge_index)

        if edge_attr is not None:
            assert isinstance(edge_attr, dict)
            edge_attr = {
                k: trim_edge_feat(v, layer, num_sampled_edges_per_hop[k])
                for k, v in edge_attr.items()
            }
            edge_attr = filter_empty_entries(edge_attr)

        #xr = trimmed_x_feat_view(x, layer, num_sampled_nodes_per_hop[k[-1]]) <== TODO: review this - for the hetero maybe necessary iterate??
        xr = {
            k: trimmed_x_feat_view(v, layer, num_sampled_nodes_per_hop[k])
            for k, v in x.items()
        }
        return x, xr, edge_index, edge_attr

    assert isinstance(num_sampled_nodes_per_hop, list)

    assert isinstance(x, Tensor)

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
        edge_attr = trim_edge_feat(edge_attr, layer, num_sampled_edges_per_hop)

    xr = trimmed_x_feat_view(x, layer, num_sampled_nodes_per_hop)
    return x, xr, edge_index, edge_attr


'''
So I assume that you mean the conv((x_src, x_dst), edge_index) is meant to replace the
conv calls (e.g. x = conv(x, edge_index))
in the forward method of the BasicGNN class.

If I understand the point here, and assuming the src/dst naming is based on the inbound/outbound
direction of edges when building neighborhood at each layer:

x_dst is a matrix num_target_nodes_for_that_layer X num_node_features, which contains the current
node representations for the target nodes only.

x_src is a matrix that might include x_dst (in the homo case for example) whose dimensions are
num_nodes_in_input_at_that_layer X num_node_features, and contains the representations of all
the nodes needed as input for the current layer.

With each conv working with a pair of tensors as input, we will require the trimming part to
return -besides the rectangular adj_matrix correctly sized at each layer- the trimmed and
non-trimmed node features matrices, in contrast to returning only the non-trimmed as it does now.

Looking at the sage_conv class implementation, the node representation tuple as input should contain:
x[0] = non-trimmed node features matrix (used as input for message passing) ==> x_src
x[1] = trimmed node features matrix (used as input for linear projection whose result will be
added to the message passing output) ==> x_dst

So with my definitions above, I guess the conv call should look like: conv((x_src, x_dst), edge_index),
which matches your suggestion.
'''


class TrimToLayer(torch.nn.Module):
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


def trim_edge_feat(trim_edge_feat: Tensor, layer: int,
                   num_sampled_edges_per_hop: List[int]) -> Tensor:
    if layer <= 0:
        return trim_edge_feat

    return trim_edge_feat.narrow(
        dim=0,
        start=0,
        length=trim_edge_feat.size(0) - num_sampled_edges_per_hop[-layer],
    )


def trimmed_x_feat_view(x: Tensor, layer: int,
                        num_sampled_nodes_per_hop: List[int]) -> Tensor:
    return x.narrow(
        dim=0,
        start=0,
        length=x.size(0) -
        num_sampled_nodes_per_hop[-(layer +
                                    1)],  # TODO: check this works in all cases
    )


'''
narrow returns a view on the object:
>>> import torch
>>> T = torch.Tensor([1,2,3,4,5])
>>> T.narrow(dim=0, start=0, length=3)
tensor([1., 2., 3.])
>>> T
tensor([1., 2., 3., 4., 5.])
>>> R = T.narrow(dim=0, start=0, length=3)
>>> R.storage().data_ptr()
94374028103872
>>> T.storage().data_ptr()
94374028103872
>>>
'''


def trim_adj(
    edge_index: Adj,
    layer: int,
    num_sampled_src_nodes_per_hop: List[int],
    num_sampled_dst_nodes_per_hop: List[int],
    num_sampled_edges_per_hop: List[int],
) -> Adj:

    # trim_adj must work for hetero case as well, so the assumption that at the first layer
    # edge_index or adj_t are square cannot hold, and we need to test a condition on the layer

    if isinstance(edge_index, Tensor):
        return edge_index.narrow(
            dim=1,
            start=0,
            length=edge_index.size(1) - num_sampled_edges_per_hop[-layer],
        )

    elif isinstance(edge_index, SparseTensor):
        # src and dst are referred to the direction of the edges, relevant for hetero
        if layer == 0:
            size = (
                edge_index.size(0) -
                num_sampled_dst_nodes_per_hop[-(layer + 1)],
                edge_index.size(
                    1
                ),  # in homo case the layer 0 edge_index is still square, so it could be size(0), but for hetero case that's not true
            )
        else:
            size = (
                edge_index.size(0) -
                num_sampled_dst_nodes_per_hop[-(layer + 1)],
                edge_index.size(1) - num_sampled_src_nodes_per_hop[-layer],
            )
        # size = (
        #     edge_index.size(0) - num_sampled_dst_nodes_per_hop[-(layer+1)],
        #     edge_index.size(0))
        # compact way to express the change in size for each layer, including layer 0, for homo case

        num_seed_nodes = size[0]

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
    nnz = rowptr[-1]
    col = torch.narrow(col, 0, 0, nnz)

    if value is not None:
        value = torch.narrow(value, 0, 0, nnz)

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
