from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor

from torch_geometric.typing import EdgeType, NodeType


def trim_to_layer(
    layer: int,
    edge_index: Tensor,
    node_attrs: Union[Tensor, Dict[NodeType, Tensor]],
    num_nodes_per_layer: Union[List[int], Dict[NodeType, List[int]]],
    num_edges_per_layer: Union[List[int], Dict[EdgeType, List[int]]],
    edge_attrs: Optional[Union[Tensor, Dict[EdgeType, Tensor]]] = None,
) -> Tuple[Union[Tensor, Dict[NodeType, Tensor]], Union[Tensor, Dict[
        EdgeType, Tensor]], Optional[Union[Tensor, Dict[EdgeType, Tensor]]]]:
    """
        Args:
        layer (int):
            Current performed layer of the overall n GNN layers in model
        node_attrs (torch.Tensor or Dict[NodeType, torch.Tensor]):
            Node feature matrix
        edge_attrs (torch.Tensor or Dict[NodeType, torch.Tensor], optional):
            Edge feature matrix
        edge_index (torch.Tensor or torch_sparse.SparseTensor):
            Adjacency matrix
        num_edges_per_layer (List[int] or Dict[EdgeType, List[int]]):
            Amount of total sampled edges for each layer
        num_nodes_per_layer (List[int] or Dict[NodeType, List[int]]):
            Amount of total sampled nodes for each layer
    """
    if layer == 0:
        return (node_attrs, edge_index,
                edge_attrs) if edge_attrs is not None else (node_attrs,
                                                            edge_index)
    else:
        if isinstance(edge_index, dict):
            if isinstance(node_attrs, dict):
                for key in node_attrs.keys():
                    node_attrs[key] = torch.narrow(
                        node_attrs[key], 0, 0, node_attrs[key].shape[0] -
                        num_nodes_per_layer[key][-layer])
            if edge_attrs is not None and isinstance(edge_attrs, dict):
                for key in edge_attrs.keys():
                    edge_attrs[key] = torch.narrow(
                        edge_attrs[key], 0, 0, edge_attrs[key].shape[0] -
                        num_edges_per_layer[key][-layer])
            if isinstance(next(iter(edge_index.values())), Tensor):
                for key in edge_index.keys():
                    edge_index[key] = torch.narrow(
                        edge_index[key], 1, 0, edge_index[key].shape[1] -
                        num_edges_per_layer[key][-layer])
        else:
            node_attrs = torch.narrow(
                node_attrs, 0, 0,
                node_attrs.shape[0] - num_nodes_per_layer[-layer])
            if edge_attrs is not None:
                edge_attrs = torch.narrow(
                    edge_attrs, 0, 0,
                    edge_attrs.shape[0] - num_edges_per_layer[-layer])
            if isinstance(edge_index, Tensor):
                edge_index = torch.narrow(
                    edge_index, 1, 0,
                    edge_index.shape[1] - num_edges_per_layer[-layer])

        return (node_attrs, edge_index,
                edge_attrs) if edge_attrs is not None else (node_attrs,
                                                            edge_index)
