from typing import Dict, Optional

import torch
from torch import Tensor

from torch_geometric.data import HeteroData
from torch_geometric.distributed import LocalFeatureStore, LocalGraphStore


def filter_dist_store(
    feature_store: LocalFeatureStore,
    graph_store: LocalGraphStore,
    node_dict: Dict[str, Tensor],
    row_dict: Dict[str, Tensor],
    col_dict: Dict[str, Tensor],
    edge_dict: Dict[str, Optional[Tensor]],
    custom_cls: Optional[HeteroData] = None,
    meta: Optional[Dict[str, Tensor]] = None,
) -> HeteroData:
    r"""Constructs a :class:`HeteroData` object from a feature store that only
    holds nodes in `node` end edges in `edge` for each node and edge type,
    respectively. Sorted attribute values are provided as metadata from
    :class:`DistNeighborSampler`."""
    # Construct a new `HeteroData` object:
    data = custom_cls() if custom_cls is not None else HeteroData()
    nfeats, nlabels, efeats = meta[-3:]

    # Filter edge storage:
    required_edge_attrs = []
    for attr in graph_store.get_all_edge_attrs():
        key = attr.edge_type
        if key in row_dict and key in col_dict:
            required_edge_attrs.append(attr)
            edge_index = torch.stack([row_dict[key], col_dict[key]], dim=0)
            data[attr.edge_type].edge_index = edge_index

    # Filter node storage:
    required_node_attrs = []
    for attr in feature_store.get_all_tensor_attrs():
        if attr.group_name in node_dict:
            attr.index = node_dict[attr.group_name]
            required_node_attrs.append(attr)
            data[attr.group_name].num_nodes = attr.index.size(0)

    if nfeats is not None:
        for attr in required_node_attrs:
            if nfeats[attr.group_name] is not None:
                data[attr.group_name][attr.attr_name] = nfeats[attr.group_name]

    if efeats is not None:
        for attr in required_edge_attrs:
            if efeats[attr.edge_type] is not None:
                data[attr.edge_type].edge_attr = efeats[attr.edge_type]

    for label in nlabels:
        data[label].y = nlabels[label]

    return data
