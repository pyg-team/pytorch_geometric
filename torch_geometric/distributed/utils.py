from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from torch import Tensor

from torch_geometric.data import HeteroData
from torch_geometric.distributed import LocalFeatureStore, LocalGraphStore
from torch_geometric.sampler import SamplerOutput
from torch_geometric.typing import Dict, NodeType, Optional


@dataclass
class NodeDict:
    r"""Class used during hetero sampling. It contains fields that refer to
    NodeDict in three situations:
    1) the nodes are to serve as srcs in the next layer,
    2) nodes with duplicates, that are further needed to create COO output
       matrix,
    3) output nodes without duplicates.
    """
    def __init__(self, node_types):
        self.src: Dict[NodeType, Tensor] = {}
        self.with_dupl: Dict[NodeType, Tensor] = {}
        self.out: Dict[NodeType, np.array] = {}

        for ntype in node_types:
            self.src.update({ntype: torch.empty(0, dtype=torch.int64)})
            self.with_dupl.update({ntype: torch.empty(0, dtype=torch.int64)})
            self.out.update({ntype: np.empty(0, dtype=np.int64)})


@dataclass
class BatchDict:
    r"""Class used during disjoint hetero sampling. It contains fields that
    refer to BatchDict in three situations:
    1) the batch is to serve as initial subgraph ids for src nodes in the next
       layer,
    2) subgraph ids with duplicates, that are further needed to create COO
       output matrix,
    3) output subgraph ids without duplicates.
    """
    def __init__(self, node_types, disjoint):
        self.src: Dict[NodeType, Tensor] = {}
        self.with_dupl: Dict[NodeType, Tensor] = {}
        self.out: Dict[NodeType, Tensor] = {}
        self.disjoint = disjoint

        if self.disjoint:
            for ntype in node_types:
                self.src.update({ntype: torch.empty(0, dtype=torch.int64)})
                self.with_dupl.update(
                    {ntype: torch.empty(0, dtype=torch.int64)})
                self.out.update({ntype: np.empty(0, dtype=np.int64)})
        else:
            for ntype in node_types:
                self.src.update({ntype: None})
                self.with_dupl.update({ntype: None})
                self.out.update({ntype: None})


def remove_duplicates(
    out: SamplerOutput,
    node: Tensor,
    batch: Optional[Tensor] = None,
    disjoint: bool = False,
) -> Tuple[Tensor, Tensor, Optional[Tensor], Optional[Tensor]]:

    num_nodes = node.numel()
    node_combined = torch.cat([node, out.node])

    if not disjoint:
        _, idx = np.unique(node_combined.cpu().numpy(), return_index=True)
        idx = torch.from_numpy(idx).to(node.device).sort().values

        node = node_combined[idx]
        src = node[num_nodes:]

        return (src, node, None, None)

    else:
        batch_combined = torch.cat([batch, out.batch])
        node_batch = torch.stack([batch_combined, node_combined], dim=0)

        _, idx = np.unique(node_batch.cpu().numpy(), axis=1, return_index=True)
        idx = torch.from_numpy(idx).to(node.device).sort().values

        batch = batch_combined[idx]
        node = node_combined[idx]
        src_batch = batch[num_nodes:]
        src = node[num_nodes:]

        return (src, node, src_batch, batch)


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
