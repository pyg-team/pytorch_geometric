from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor

from torch_geometric.data import HeteroData
from torch_geometric.distributed.local_feature_store import LocalFeatureStore
from torch_geometric.distributed.local_graph_store import LocalGraphStore
from torch_geometric.sampler import SamplerOutput
from torch_geometric.typing import EdgeType, NodeType


@dataclass
class DistEdgeHeteroSamplerInput:
    r"""The sampling input of
    :meth:`~torch_geometric.dstributed.DistNeighborSampler.node_sample` used
    during distributed heterogeneous link sampling when source and target node
    types of an input edge are different.

    Args:
        input_id (torch.Tensor, optional): The indices of the data loader input
            of the current mini-batch.
        node_dict (Dict[NodeType, torch.Tensor]): The indices of seed nodes of
            a given node types to start sampling from.
        time_dict (Dict[NodeType, torch.Tensor], optional): The timestamp for
            the seed nodes of a given node types. (default: :obj:`None`)
        input_type (str, optional): The input node type. (default: :obj:`None`)
    """
    input_id: Optional[Tensor]
    node_dict: Dict[NodeType, Tensor]
    time_dict: Optional[Dict[NodeType, Tensor]] = None
    input_type: Optional[EdgeType] = None


class NodeDict:
    r"""Class used during heterogeneous sampling.
    1) The nodes to serve as source nodes in the next layer.
    2) The nodes with duplicates that are further needed to create COO output.
    3) The output nodes without duplicates.
    """
    def __init__(self, node_types, num_hops):
        self.src: Dict[NodeType, List[Tensor]] = {
            k: (num_hops + 1) * [torch.empty(0, dtype=torch.int64)]
            for k in node_types
        }
        self.with_dupl: Dict[NodeType, Tensor] = {
            k: torch.empty(0, dtype=torch.int64)
            for k in node_types
        }
        self.out: Dict[NodeType, Tensor] = {
            k: torch.empty(0, dtype=torch.int64)
            for k in node_types
        }
        self.seed_time: Dict[NodeType, List[Tensor]] = {
            k: num_hops * [torch.empty(0, dtype=torch.int64)]
            for k in node_types
        }


class BatchDict:
    r"""Class used during disjoint heterogeneous sampling.
    1) The batch to serve as initial subgraph IDs for source nodes in the next
       layer.
    2) The subgraph IDs with duplicates that are further needed to create COO
       output.
    3) The output subgraph IDs without duplicates.
    """
    def __init__(self, node_types, num_hops):
        self.src: Dict[NodeType, List[Tensor]] = {
            k: (num_hops + 1) * [torch.empty(0, dtype=torch.int64)]
            for k in node_types
        }
        self.with_dupl: Dict[NodeType, Tensor] = {
            k: torch.empty(0, dtype=torch.int64)
            for k in node_types
        }
        self.out: Dict[NodeType, Tensor] = {
            k: torch.empty(0, dtype=torch.int64)
            for k in node_types
        }


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
    input_type: str = None,
) -> HeteroData:
    r"""Constructs a :class:`HeteroData` object from a feature store that only
    holds nodes in `node` end edges in `edge` for each node and edge type,
    respectively. Sorted attribute values are provided as metadata from
    :class:`DistNeighborSampler`.
    """
    # Construct a new `HeteroData` object:
    data = custom_cls() if custom_cls is not None else HeteroData()
    nfeats, labels, efeats = meta[-3:]

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

    if nfeats:
        for attr in required_node_attrs:
            if nfeats[attr.group_name] is not None:
                data[attr.group_name][attr.attr_name] = nfeats[attr.group_name]

    if efeats:
        for attr in required_edge_attrs:
            if efeats[attr.edge_type] is not None:
                data[attr.edge_type].edge_attr = efeats[attr.edge_type]

    if labels:
        data[input_type].y = labels[input_type]

    return data


def as_str(inputs: Union[NodeType, EdgeType]) -> str:
    if isinstance(inputs, NodeType):
        return inputs
    elif isinstance(inputs, (list, tuple)) and len(inputs) == 3:
        return '__'.join(inputs)
    return ''


def reverse_edge_type(etype: EdgeType) -> EdgeType:
    src, rel, dst = etype
    if src != dst:
        if rel.split('_', 1)[0] == 'rev':
            # undirected edge with `rev_` prefix.
            rel = rel.split('_', 1)[1]
        else:
            rel = 'rev_' + rel

    return dst, rel, src
