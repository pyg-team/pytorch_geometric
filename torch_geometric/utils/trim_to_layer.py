from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor

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

    x = x.narrow(
        dim=0,
        start=0,
        length=x.size(0) - num_sampled_nodes_per_hop[-layer],
    )
    edge_index = edge_index.narrow(
        dim=1,
        start=0,
        length=edge_index.size(1) - num_sampled_edges_per_hop[-layer],
    )
    if edge_attr is not None:
        edge_attr = edge_attr.narrow(
            dim=0,
            start=0,
            length=edge_attr.size(0) - num_sampled_edges_per_hop[-layer],
        )
    return x, edge_index, edge_attr


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
