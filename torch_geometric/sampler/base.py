import copy
import math
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

import torch
from torch import Tensor

from torch_geometric.data import Data, FeatureStore, GraphStore, HeteroData
from torch_geometric.sampler.utils import (
    global_to_local_node_idx,
    local_to_global_node_idx,
    to_bidirectional,
    unique_unsorted,
)
from torch_geometric.typing import EdgeType, EdgeTypeStr, NodeType, OptTensor
from torch_geometric.utils.mixin import CastMixin


class DataType(Enum):
    r"""The data type a sampler is operating on."""
    homogeneous = 'homogeneous'
    heterogeneous = 'heterogeneous'
    remote = 'remote'

    @classmethod
    def from_data(cls, data: Any):
        if isinstance(data, Data):
            return cls.homogeneous
        elif isinstance(data, HeteroData):
            return cls.heterogeneous
        elif (isinstance(data, (list, tuple)) and len(data) == 2
              and isinstance(data[0], FeatureStore)
              and isinstance(data[1], GraphStore)):
            return cls.remote

        raise ValueError(f"Expected a 'Data', 'HeteroData', or a tuple of "
                         f"'FeatureStore' and 'GraphStore' "
                         f"(got '{type(data)}')")


class SubgraphType(Enum):
    r"""The type of the returned subgraph."""
    directional = 'directional'
    bidirectional = 'bidirectional'
    induced = 'induced'


@dataclass(init=False)
class NodeSamplerInput(CastMixin):
    r"""The sampling input of
    :meth:`~torch_geometric.sampler.BaseSampler.sample_from_nodes`.

    Args:
        input_id (torch.Tensor, optional): The indices of the data loader input
            of the current mini-batch.
        node (torch.Tensor): The indices of seed nodes to start sampling from.
        time (torch.Tensor, optional): The timestamp for the seed nodes.
            (default: :obj:`None`)
        input_type (str, optional): The input node type (in case of sampling in
            a heterogeneous graph). (default: :obj:`None`)
    """
    input_id: OptTensor
    node: Tensor
    time: OptTensor = None
    input_type: Optional[NodeType] = None

    def __init__(
        self,
        input_id: OptTensor,
        node: Tensor,
        time: OptTensor = None,
        input_type: Optional[NodeType] = None,
    ):
        if input_id is not None:
            input_id = input_id.cpu()
        node = node.cpu()
        if time is not None:
            time = time.cpu()

        self.input_id = input_id
        self.node = node
        self.time = time
        self.input_type = input_type

    def __getitem__(self, index: Union[Tensor, Any]) -> 'NodeSamplerInput':
        if not isinstance(index, Tensor):
            index = torch.tensor(index, dtype=torch.long)

        return NodeSamplerInput(
            self.input_id[index] if self.input_id is not None else index,
            self.node[index],
            self.time[index] if self.time is not None else None,
            self.input_type,
        )


@dataclass(init=False)
class EdgeSamplerInput(CastMixin):
    r"""The sampling input of
    :meth:`~torch_geometric.sampler.BaseSampler.sample_from_edges`.

    Args:
        input_id (torch.Tensor, optional): The indices of the data loader input
            of the current mini-batch.
        row (torch.Tensor): The source node indices of seed links to start
            sampling from.
        col (torch.Tensor): The destination node indices of seed links to start
            sampling from.
        label (torch.Tensor, optional): The label for the seed links.
            (default: :obj:`None`)
        time (torch.Tensor, optional): The timestamp for the seed links.
            (default: :obj:`None`)
        input_type (Tuple[str, str, str], optional): The input edge type (in
            case of sampling in a heterogeneous graph). (default: :obj:`None`)
    """
    input_id: OptTensor
    row: Tensor
    col: Tensor
    label: OptTensor = None
    time: OptTensor = None
    input_type: Optional[EdgeType] = None

    def __init__(
        self,
        input_id: OptTensor,
        row: Tensor,
        col: Tensor,
        label: OptTensor = None,
        time: OptTensor = None,
        input_type: Optional[EdgeType] = None,
    ):
        if input_id is not None:
            input_id = input_id.cpu()
        row = row.clone().cpu()
        col = col.clone().cpu()
        if label is not None:
            label = label.cpu()
        if time is not None:
            time = time.cpu()

        self.input_id = input_id
        self.row = row
        self.col = col
        self.label = label
        self.time = time
        self.input_type = input_type

    def __getitem__(self, index: Union[Tensor, Any]) -> 'EdgeSamplerInput':
        if not isinstance(index, Tensor):
            index = torch.tensor(index, dtype=torch.long)

        return EdgeSamplerInput(
            self.input_id[index] if self.input_id is not None else index,
            self.row[index],
            self.col[index],
            self.label[index] if self.label is not None else None,
            self.time[index] if self.time is not None else None,
            self.input_type,
        )


@dataclass
class SamplerOutput(CastMixin):
    r"""The sampling output of a :class:`~torch_geometric.sampler.BaseSampler`
    on homogeneous graphs.

    Args:
        node (torch.Tensor): The sampled nodes in the original graph.
        row (torch.Tensor): The source node indices of the sampled subgraph.
            Indices must be re-indexed to :obj:`{ 0, ..., num_nodes - 1 }`
            corresponding to the nodes in the :obj:`node` tensor.
        col (torch.Tensor): The destination node indices of the sampled
            subgraph.
            Indices must be re-indexed to :obj:`{ 0, ..., num_nodes - 1 }`
            corresponding to the nodes in the :obj:`node` tensor.
        edge (torch.Tensor, optional): The sampled edges in the original graph.
            This tensor is used to obtain edge features from the original
            graph. If no edge attributes are present, it may be omitted.
        batch (torch.Tensor, optional): The vector to identify the seed node
            for each sampled node. Can be present in case of disjoint subgraph
            sampling per seed node. (default: :obj:`None`)
        num_sampled_nodes (List[int], optional): The number of sampled nodes
            per hop. (default: :obj:`None`)
        num_sampled_edges (List[int], optional): The number of sampled edges
            per hop. (default: :obj:`None`)
        orig_row (torch.Tensor, optional): The original source node indices
            returned by the sampler.
            Filled in case :meth:`to_bidirectional` is called with the
            :obj:`keep_orig_edges` option. (default: :obj:`None`)
        orig_col (torch.Tensor, optional): The original destination node
            indices indices returned by the sampler.
            Filled in case :meth:`to_bidirectional` is called with the
            :obj:`keep_orig_edges` option. (default: :obj:`None`)
        metadata: (Any, optional): Additional metadata information.
            (default: :obj:`None`)
    """
    node: Tensor
    row: Tensor
    col: Tensor
    edge: OptTensor
    batch: OptTensor = None
    num_sampled_nodes: Optional[List[int]] = None
    num_sampled_edges: Optional[List[int]] = None
    orig_row: Tensor = None
    orig_col: Tensor = None
    # TODO(manan): refine this further; it does not currently define a proper
    # API for the expected output of a sampler.
    metadata: Optional[Any] = None
    _seed_node: OptTensor = field(repr=False, default=None)

    @property
    def global_row(self) -> Tensor:
        return local_to_global_node_idx(self.node, self.row)

    @property
    def global_col(self) -> Tensor:
        return local_to_global_node_idx(self.node, self.col)

    @property
    def seed_node(self) -> Tensor:
        # can be set manually if the seed nodes are not contained in the
        # sampled nodes
        if self._seed_node is None:
            self._seed_node = local_to_global_node_idx(
                self.node, self.batch) if self.batch is not None else None
        return self._seed_node

    @seed_node.setter
    def seed_node(self, value: Tensor):
        assert len(value) == len(self.node)
        self._seed_node = value

    @property
    def global_orig_row(self) -> Tensor:
        return local_to_global_node_idx(
            self.node, self.orig_row) if self.orig_row is not None else None

    @property
    def global_orig_col(self) -> Tensor:
        return local_to_global_node_idx(
            self.node, self.orig_col) if self.orig_col is not None else None

    def to_bidirectional(
        self,
        keep_orig_edges: bool = False,
    ) -> 'SamplerOutput':
        r"""Converts the sampled subgraph into a bidirectional variant, in
        which all sampled edges are guaranteed to be bidirectional.

        Args:
            keep_orig_edges (bool, optional): If specified, directional edges
                are still maintained. (default: :obj:`False`)
        """
        out = copy.copy(self)

        if keep_orig_edges:
            out.orig_row = self.row
            out.orig_col = self.col
        else:
            out.num_sampled_nodes = out.num_sampled_edges = None

        out.row, out.col, out.edge = to_bidirectional(
            row=self.row,
            col=self.col,
            rev_row=self.row,
            rev_col=self.col,
            edge_id=self.edge,
            rev_edge_id=self.edge,
        )

        return out

    @classmethod
    def collate(cls, outputs: List['SamplerOutput'],
                replace: bool = True) -> 'SamplerOutput':
        r"""Collate a list of :class:`~torch_geometric.sampler.SamplerOutput`
        objects into a single :class:`~torch_geometric.sampler.SamplerOutput`
        object. Requires that they all have the same fields.
        """
        if len(outputs) == 0:
            raise ValueError("Cannot collate an empty list of SamplerOutputs")
        out = outputs[0]
        has_edge = out.edge is not None
        has_orig_row = out.orig_row is not None
        has_orig_col = out.orig_col is not None
        has_batch = out.batch is not None
        has_num_sampled_nodes = out.num_sampled_nodes is not None
        has_num_sampled_edges = out.num_sampled_edges is not None

        try:
            for i, sample_output in enumerate(outputs):  # noqa
                assert not has_edge == (sample_output.edge is None)
                assert not has_orig_row == (sample_output.orig_row is None)
                assert not has_orig_col == (sample_output.orig_col is None)
                assert not has_batch == (sample_output.batch is None)
                assert not has_num_sampled_nodes == (
                    sample_output.num_sampled_nodes is None)
                assert not has_num_sampled_edges == (
                    sample_output.num_sampled_edges is None)
        except AssertionError:
            error_str = f"Output {i+1} has a different field than the first output"  # noqa
            raise ValueError(error_str)  # noqa

        for other in outputs[1:]:
            out = out.merge_with(other, replace=replace)
        return out

    def merge_with(self, other: 'SamplerOutput',
                   replace: bool = True) -> 'SamplerOutput':
        """Merges two SamplerOutputs.
        If replace is True, self's nodes and edges take precedence.
        """
        if not replace:
            return SamplerOutput(
                node=torch.cat([self.node, other.node], dim=0),
                row=torch.cat([self.row, len(self.node) + other.row], dim=0),
                col=torch.cat([self.col, len(self.node) + other.col], dim=0),
                edge=torch.cat([self.edge, other.edge], dim=0)
                if self.edge is not None and other.edge is not None else None,
                batch=torch.cat(
                    [self.batch, len(self.node) + other.batch], dim=0) if
                self.batch is not None and other.batch is not None else None,
                num_sampled_nodes=self.num_sampled_nodes +
                other.num_sampled_nodes if self.num_sampled_nodes is not None
                and other.num_sampled_nodes is not None else None,
                num_sampled_edges=self.num_sampled_edges +
                other.num_sampled_edges if self.num_sampled_edges is not None
                and other.num_sampled_edges is not None else None,
                orig_row=torch.cat(
                    [self.orig_row,
                     len(self.node) +
                     other.orig_row], dim=0) if self.orig_row is not None
                and other.orig_row is not None else None,
                orig_col=torch.cat(
                    [self.orig_col,
                     len(self.node) +
                     other.orig_col], dim=0) if self.orig_col is not None
                and other.orig_col is not None else None,
                metadata=[self.metadata, other.metadata],
            )
        else:

            # NODES
            old_nodes, new_nodes = self.node, other.node
            old_node_uid, new_node_uid = [old_nodes], [new_nodes]

            # batch tracks disjoint subgraph samplings
            if self.batch is not None and other.batch is not None:
                # Transform the batch indices to be global node ids
                old_batch_nodes = self.seed_node
                new_batch_nodes = other.seed_node
                old_node_uid.append(old_batch_nodes)
                new_node_uid.append(new_batch_nodes)

            # NOTE: if any new node fields are added,
            # they need to be merged here

            old_node_uid = torch.stack(old_node_uid, dim=1)
            new_node_uid = torch.stack(new_node_uid, dim=1)

            merged_node_uid = unique_unsorted(
                torch.cat([old_node_uid, new_node_uid], dim=0))
            num_old_nodes = old_node_uid.shape[0]

            # Recompute num sampled nodes for second output,
            # subtracting out nodes already seen in first output
            merged_node_num_sampled_nodes = None
            if (self.num_sampled_nodes is not None
                    and other.num_sampled_nodes is not None):
                merged_node_num_sampled_nodes = copy.copy(
                    self.num_sampled_nodes)
                curr_index = 0
                # NOTE: There's an assumption here that no two nodes will be
                # sampled twice in the same SampleOutput object
                for minibatch in other.num_sampled_nodes:
                    size_of_intersect = torch.cat([
                        old_node_uid,
                        new_node_uid[curr_index:curr_index + minibatch]
                    ]).unique(dim=0, sorted=False).shape[0] - num_old_nodes
                    merged_node_num_sampled_nodes.append(size_of_intersect)
                    curr_index += minibatch

            merged_nodes = merged_node_uid[:, 0]
            merged_batch = None
            if self.batch is not None and other.batch is not None:
                # Restore the batch indices to be relative to the nodes field
                ref_merged_batch_nodes = merged_node_uid[:, 1].unsqueeze(
                    -1).expand(-1, 2)  # num_nodes x 2
                merged_batch = global_to_local_node_idx(
                    merged_node_uid, ref_merged_batch_nodes)

            # EDGES
            is_bidirectional = self.orig_row is not None \
                and self.orig_col is not None \
                and other.orig_row is not None \
                and other.orig_col is not None
            if is_bidirectional:
                old_row, old_col = self.orig_row, self.orig_col
                new_row, new_col = other.orig_row, other.orig_col
            else:
                old_row, old_col = self.row, self.col
                new_row, new_col = other.row, other.col

            # Transform the row and col indices to be global node ids
            # instead of relative indices to nodes field
            # Edge uids build off of node uids
            old_row_idx, old_col_idx = local_to_global_node_idx(
                old_node_uid,
                old_row), local_to_global_node_idx(old_node_uid, old_col)
            new_row_idx, new_col_idx = local_to_global_node_idx(
                new_node_uid,
                new_row), local_to_global_node_idx(new_node_uid, new_col)

            old_edge_uid, new_edge_uid = [old_row_idx, old_col_idx
                                          ], [new_row_idx, new_col_idx]

            row_idx = 0
            col_idx = old_row_idx.shape[1]
            edge_idx = old_row_idx.shape[1] + old_col_idx.shape[1]

            if self.edge is not None and other.edge is not None:
                if is_bidirectional:
                    # bidirectional duplicates edge ids
                    old_edge_uid_ref = torch.stack([self.row, self.col],
                                                   dim=1)  # num_edges x 2
                    old_orig_edge_uid_ref = torch.stack(
                        [self.orig_row, self.orig_col],
                        dim=1)  # num_orig_edges x 2

                    old_edge_idx = global_to_local_node_idx(
                        old_edge_uid_ref, old_orig_edge_uid_ref)
                    old_edge = self.edge[old_edge_idx]

                    new_edge_uid_ref = torch.stack([other.row, other.col],
                                                   dim=1)  # num_edges x 2
                    new_orig_edge_uid_ref = torch.stack(
                        [other.orig_row, other.orig_col],
                        dim=1)  # num_orig_edges x 2

                    new_edge_idx = global_to_local_node_idx(
                        new_edge_uid_ref, new_orig_edge_uid_ref)
                    new_edge = other.edge[new_edge_idx]

                else:
                    old_edge, new_edge = self.edge, other.edge

                old_edge_uid.append(old_edge.unsqueeze(-1))
                new_edge_uid.append(new_edge.unsqueeze(-1))

            old_edge_uid = torch.cat(old_edge_uid, dim=1)
            new_edge_uid = torch.cat(new_edge_uid, dim=1)

            merged_edge_uid = unique_unsorted(
                torch.cat([old_edge_uid, new_edge_uid], dim=0))
            num_old_edges = old_edge_uid.shape[0]

            merged_edge_num_sampled_edges = None
            if (self.num_sampled_edges is not None
                    and other.num_sampled_edges is not None):
                merged_edge_num_sampled_edges = copy.copy(
                    self.num_sampled_edges)
                curr_index = 0
                # NOTE: There's an assumption here that no two edges will be
                # sampled twice in the same SampleOutput object
                for minibatch in other.num_sampled_edges:
                    size_of_intersect = torch.cat([
                        old_edge_uid,
                        new_edge_uid[curr_index:curr_index + minibatch]
                    ]).unique(dim=0, sorted=False).shape[0] - num_old_edges
                    merged_edge_num_sampled_edges.append(size_of_intersect)
                    curr_index += minibatch

            merged_row = merged_edge_uid[:, row_idx:col_idx]
            merged_col = merged_edge_uid[:, col_idx:edge_idx]
            merged_edge = merged_edge_uid[:, edge_idx:].squeeze() \
                if self.edge is not None and other.edge is not None else None

            # restore to row and col indices relative to nodes field
            merged_row = global_to_local_node_idx(merged_node_uid, merged_row)
            merged_col = global_to_local_node_idx(merged_node_uid, merged_col)

            out = SamplerOutput(
                node=merged_nodes,
                row=merged_row,
                col=merged_col,
                edge=merged_edge,
                batch=merged_batch,
                num_sampled_nodes=merged_node_num_sampled_nodes,
                num_sampled_edges=merged_edge_num_sampled_edges,
                metadata=[self.metadata, other.metadata],
            )
            # Restores orig_row and orig_col if they existed before merging
            if is_bidirectional:
                out = out.to_bidirectional(keep_orig_edges=True)
            return out


@dataclass
class HeteroSamplerOutput(CastMixin):
    r"""The sampling output of a :class:`~torch_geometric.sampler.BaseSampler`
    on heterogeneous graphs.

    Args:
        node (Dict[str, torch.Tensor]): The sampled nodes in the original graph
            for each node type.
        row (Dict[Tuple[str, str, str], torch.Tensor]): The source node indices
            of the sampled subgraph for each edge type.
            Indices must be re-indexed to :obj:`{ 0, ..., num_nodes - 1 }`
            corresponding to the nodes in the :obj:`node` tensor of the source
            node type.
        col (Dict[Tuple[str, str, str], torch.Tensor]): The destination node
            indices of the sampled subgraph for each edge type.
            Indices must be re-indexed to :obj:`{ 0, ..., num_nodes - 1 }`
            corresponding to the nodes in the :obj:`node` tensor of the
            destination node type.
        edge (Dict[Tuple[str, str, str], torch.Tensor], optional): The sampled
            edges in the original graph for each edge type.
            This tensor is used to obtain edge features from the original
            graph. If no edge attributes are present, it may be omitted.
        batch (Dict[str, torch.Tensor], optional): The vector to identify the
            seed node for each sampled node for each node type. Can be present
            in case of disjoint subgraph sampling per seed node.
            (default: :obj:`None`)
        num_sampled_nodes (Dict[str, List[int]], optional): The number of
            sampled nodes for each node type and each layer.
            (default: :obj:`None`)
        num_sampled_edges (Dict[EdgeType, List[int]], optional): The number of
            sampled edges for each edge type and each layer.
            (default: :obj:`None`)
        orig_row (Dict[EdgeType, torch.Tensor], optional): The original source
            node indices returned by the sampler.
            Filled in case :meth:`to_bidirectional` is called with the
            :obj:`keep_orig_edges` option. (default: :obj:`None`)
        orig_col (Dict[EdgeType, torch.Tensor], optional): The original
            destination node indices returned by the sampler.
            Filled in case :meth:`to_bidirectional` is called with the
            :obj:`keep_orig_edges` option. (default: :obj:`None`)
        metadata: (Any, optional): Additional metadata information.
            (default: :obj:`None`)
    """
    node: Dict[NodeType, Tensor]
    row: Dict[EdgeType, Tensor]
    col: Dict[EdgeType, Tensor]
    edge: Dict[EdgeType, OptTensor]
    batch: Optional[Dict[NodeType, Tensor]] = None
    num_sampled_nodes: Optional[Dict[NodeType, List[int]]] = None
    num_sampled_edges: Optional[Dict[EdgeType, List[int]]] = None
    orig_row: Optional[Dict[EdgeType, Tensor]] = None
    orig_col: Optional[Dict[EdgeType, Tensor]] = None
    # TODO(manan): refine this further; it does not currently define a proper
    # API for the expected output of a sampler.
    metadata: Optional[Any] = None

    @property
    def global_row(self) -> Dict[EdgeType, Tensor]:
        return {
            edge_type: local_to_global_node_idx(self.node[edge_type[0]], row)
            for edge_type, row in self.row.items()
        }

    @property
    def global_col(self) -> Dict[EdgeType, Tensor]:
        return {
            edge_type: local_to_global_node_idx(self.node[edge_type[2]], col)
            for edge_type, col in self.col.items()
        }

    @property
    def seed_node(self) -> Optional[Dict[NodeType, Tensor]]:
        return {
            node_type: local_to_global_node_idx(self.node[node_type], batch)
            for node_type, batch in self.batch.items()
        } if self.batch is not None else None

    @property
    def global_orig_row(self) -> Optional[Dict[EdgeType, Tensor]]:
        return {
            edge_type: local_to_global_node_idx(self.node[edge_type[0]],
                                                orig_row)
            for edge_type, orig_row in self.orig_row.items()
        } if self.orig_row is not None else None

    @property
    def global_orig_col(self) -> Optional[Dict[EdgeType, Tensor]]:
        return {
            edge_type: local_to_global_node_idx(self.node[edge_type[2]],
                                                orig_col)
            for edge_type, orig_col in self.orig_col.items()
        } if self.orig_col is not None else None

    def to_bidirectional(
        self,
        keep_orig_edges: bool = False,
    ) -> 'SamplerOutput':
        r"""Converts the sampled subgraph into a bidirectional variant, in
        which all sampled edges are guaranteed to be bidirectional.

        Args:
            keep_orig_edges (bool, optional): If specified, directional edges
                are still maintained. (default: :obj:`False`)
        """
        out = copy.copy(self)
        out.row = copy.copy(self.row)
        out.col = copy.copy(self.col)
        out.edge = copy.copy(self.edge)

        if keep_orig_edges:
            out.orig_row = {}
            out.orig_col = {}
            for key in self.row.keys():
                out.orig_row[key] = self.row[key]
                out.orig_col[key] = self.col[key]
        else:
            out.num_sampled_nodes = out.num_sampled_edges = None

        src_dst_dict = defaultdict(list)
        edge_types = self.row.keys()
        edge_types = [k for k in edge_types if not k[1].startswith('rev_')]
        for edge_type in edge_types:
            src, rel, dst = edge_type
            rev_edge_type = (dst, f'rev_{rel}', src)

            if src == dst and rev_edge_type not in self.row:
                out.row[edge_type], out.col[edge_type], _ = to_bidirectional(
                    row=self.row[edge_type],
                    col=self.col[edge_type],
                    rev_row=self.row[edge_type],
                    rev_col=self.col[edge_type],
                )
                if out.edge is not None:
                    out.edge[edge_type] = None

            elif rev_edge_type in self.row:
                out.row[edge_type], out.col[edge_type], _ = to_bidirectional(
                    row=self.row[edge_type],
                    col=self.col[edge_type],
                    rev_row=self.row[rev_edge_type],
                    rev_col=self.col[rev_edge_type],
                )
                out.row[rev_edge_type] = out.col[edge_type]
                out.col[rev_edge_type] = out.row[edge_type]
                if out.edge is not None:
                    out.edge[edge_type] = None
                    out.edge[rev_edge_type] = None

            else:  # Find the reverse edge type (if it is unique):
                if len(src_dst_dict) == 0:  # Create mapping lazily.
                    for key in self.row.keys():
                        v1, _, v2 = key
                        src_dst_dict[(v1, v2)].append(key)

                if len(src_dst_dict[(dst, src)]) == 1:
                    rev_edge_type = src_dst_dict[(dst, src)][0]
                    row, col, _ = to_bidirectional(
                        row=self.row[edge_type],
                        col=self.col[edge_type],
                        rev_row=self.row[rev_edge_type],
                        rev_col=self.col[rev_edge_type],
                    )
                    out.row[edge_type] = row
                    out.col[edge_type] = col
                    if out.edge is not None:
                        out.edge[edge_type] = None

                else:
                    warnings.warn(
                        f"Cannot convert to bidirectional graph "
                        f"since the edge type {edge_type} does not "
                        f"seem to have a reverse edge type", stacklevel=2)

        return out

    @classmethod
    def collate(cls, outputs: List['HeteroSamplerOutput'],
                replace: bool = True) -> 'HeteroSamplerOutput':
        r"""Collate a list of
        :class:`~torch_geometric.sampler.HeteroSamplerOutput`objects into a
        single :class:`~torch_geometric.sampler.HeteroSamplerOutput` object.
        Requires that they all have the same fields.
        """
        # TODO(zaristei)
        raise NotImplementedError

    def merge_with(self, other: 'HeteroSamplerOutput',
                   replace: bool = True) -> 'HeteroSamplerOutput':
        """Merges two HeteroSamplerOutputs.
        If replace is True, self's nodes and edges take precedence.
        """
        # TODO(zaristei)
        raise NotImplementedError


@dataclass(frozen=True)
class NumNeighbors:
    r"""The number of neighbors to sample in a homogeneous or heterogeneous
    graph. In heterogeneous graphs, may also take in a dictionary denoting
    the amount of neighbors to sample for individual edge types.

    Args:
        values (List[int] or Dict[Tuple[str, str, str], List[int]]): The
            number of neighbors to sample.
            If an entry is set to :obj:`-1`, all neighbors will be included.
            In heterogeneous graphs, may also take in a dictionary denoting
            the amount of neighbors to sample for individual edge types.
        default (List[int], optional): The default number of neighbors for edge
            types not specified in :obj:`values`. (default: :obj:`None`)
    """
    values: Union[List[int], Dict[EdgeTypeStr, List[int]]]
    default: Optional[List[int]] = None

    def __init__(
        self,
        values: Union[List[int], Dict[EdgeType, List[int]]],
        default: Optional[List[int]] = None,
    ):
        if isinstance(values, (tuple, list)) and default is not None:
            raise ValueError(f"'default' must be set to 'None' in case a "
                             f"single list is given as the number of "
                             f"neighbors (got '{type(default)})'")

        if isinstance(values, dict):
            values = {EdgeTypeStr(key): value for key, value in values.items()}

        # Write to `__dict__` since dataclass is annotated with `frozen=True`:
        self.__dict__['values'] = values
        self.__dict__['default'] = default

    def _get_values(
        self,
        edge_types: Optional[List[EdgeType]] = None,
        mapped: bool = False,
    ) -> Union[List[int], Dict[Union[EdgeType, EdgeTypeStr], List[int]]]:

        if edge_types is not None:
            if isinstance(self.values, (tuple, list)):
                default = self.values
            elif isinstance(self.values, dict):
                default = self.default
            else:
                raise AssertionError()

            # Confirm that `values` only hold valid edge types:
            if isinstance(self.values, dict):
                edge_types_str = {EdgeTypeStr(key) for key in edge_types}
                invalid_edge_types = set(self.values.keys()) - edge_types_str
                if len(invalid_edge_types) > 0:
                    raise ValueError("Not all edge types specified in "
                                     "'num_neighbors' exist in the graph")

            out = {}
            for edge_type in edge_types:
                edge_type_str = EdgeTypeStr(edge_type)
                if edge_type_str in self.values:
                    out[edge_type_str if mapped else edge_type] = (
                        self.values[edge_type_str])
                else:
                    if default is None:
                        raise ValueError(f"Missing number of neighbors for "
                                         f"edge type '{edge_type}'")
                    out[edge_type_str if mapped else edge_type] = default

        elif isinstance(self.values, dict) and not mapped:
            out = {key.to_tuple(): value for key, value in self.values.items()}

        else:
            out = copy.copy(self.values)

        if isinstance(out, dict):
            num_hops = {len(v) for v in out.values()}
            if len(num_hops) > 1:
                raise ValueError(f"Number of hops must be the same across all "
                                 f"edge types (got {len(num_hops)} different "
                                 f"number of hops)")

        return out

    def get_values(
        self,
        edge_types: Optional[List[EdgeType]] = None,
    ) -> Union[List[int], Dict[EdgeType, List[int]]]:
        r"""Returns the number of neighbors.

        Args:
            edge_types (List[Tuple[str, str, str]], optional): The edge types
                to generate the number of neighbors for. (default: :obj:`None`)
        """
        if '_values' in self.__dict__:
            return self.__dict__['_values']

        values = self._get_values(edge_types, mapped=False)

        self.__dict__['_values'] = values
        return values

    def get_mapped_values(
        self,
        edge_types: Optional[List[EdgeType]] = None,
    ) -> Union[List[int], Dict[str, List[int]]]:
        r"""Returns the number of neighbors.
        For heterogeneous graphs, a dictionary is returned in which edge type
        tuples are converted to strings.

        Args:
            edge_types (List[Tuple[str, str, str]], optional): The edge types
                to generate the number of neighbors for. (default: :obj:`None`)
        """
        if '_mapped_values' in self.__dict__:
            return self.__dict__['_mapped_values']

        values = self._get_values(edge_types, mapped=True)

        self.__dict__['_mapped_values'] = values
        return values

    @property
    def num_hops(self) -> int:
        r"""Returns the number of hops."""
        if '_num_hops' in self.__dict__:
            return self.__dict__['_num_hops']

        if isinstance(self.values, (tuple, list)):
            num_hops = max(len(self.values), len(self.default or []))
        else:  # isinstance(self.values, dict):
            num_hops = max([0] + [len(v) for v in self.values.values()])
            num_hops = max(num_hops, len(self.default or []))

        self.__dict__['_num_hops'] = num_hops
        return num_hops

    def __len__(self) -> int:
        r"""Returns the number of hops."""
        return self.num_hops


class NegativeSamplingMode(Enum):
    # 'binary': Randomly sample negative edges in the graph.
    binary = 'binary'
    # 'triplet': Randomly sample negative destination nodes for each positive
    # source node.
    triplet = 'triplet'


@dataclass
class NegativeSampling(CastMixin):
    r"""The negative sampling configuration of a
    :class:`~torch_geometric.sampler.BaseSampler` when calling
    :meth:`~torch_geometric.sampler.BaseSampler.sample_from_edges`.

    Args:
        mode (str): The negative sampling mode
            (:obj:`"binary"` or :obj:`"triplet"`).
            If set to :obj:`"binary"`, will randomly sample negative links
            from the graph.
            If set to :obj:`"triplet"`, will randomly sample negative
            destination nodes for each positive source node.
        amount (int or float, optional): The ratio of sampled negative edges to
            the number of positive edges. (default: :obj:`1`)
        src_weight (torch.Tensor, optional): A node-level vector determining
            the sampling of source nodes. Does not necessarily need to sum up
            to one. If not given, negative nodes will be sampled uniformly.
            (default: :obj:`None`)
        dst_weight (torch.Tensor, optional): A node-level vector determining
            the sampling of destination nodes. Does not necessarily need to sum
            up to one. If not given, negative nodes will be sampled uniformly.
            (default: :obj:`None`)
    """
    mode: NegativeSamplingMode
    amount: Union[int, float] = 1
    src_weight: Optional[Tensor] = None
    dst_weight: Optional[Tensor] = None

    def __init__(
        self,
        mode: Union[NegativeSamplingMode, str],
        amount: Union[int, float] = 1,
        src_weight: Optional[Tensor] = None,
        dst_weight: Optional[Tensor] = None,
    ):
        self.mode = NegativeSamplingMode(mode)
        self.amount = amount
        self.src_weight = src_weight
        self.dst_weight = dst_weight

        if self.amount <= 0:
            raise ValueError(f"The attribute 'amount' needs to be positive "
                             f"for '{self.__class__.__name__}' "
                             f"(got {self.amount})")

        if self.is_triplet():
            if self.amount != math.ceil(self.amount):
                raise ValueError(f"The attribute 'amount' needs to be an "
                                 f"integer for '{self.__class__.__name__}' "
                                 f"with 'triplet' negative sampling "
                                 f"(got {self.amount}).")
            self.amount = math.ceil(self.amount)

    def is_binary(self) -> bool:
        return self.mode == NegativeSamplingMode.binary

    def is_triplet(self) -> bool:
        return self.mode == NegativeSamplingMode.triplet

    def sample(
        self,
        num_samples: int,
        endpoint: Literal['src', 'dst'],
        num_nodes: Optional[int] = None,
    ) -> Tensor:
        r"""Generates :obj:`num_samples` negative samples."""
        weight = self.src_weight if endpoint == 'src' else self.dst_weight

        if weight is None:
            if num_nodes is None:
                raise ValueError(
                    f"Cannot sample negatives in '{self.__class__.__name__}' "
                    f"without passing the 'num_nodes' argument")
            return torch.randint(num_nodes, (num_samples, ))

        if num_nodes is not None and weight.numel() != num_nodes:
            raise ValueError(
                f"The 'weight' attribute in '{self.__class__.__name__}' "
                f"needs to match the number of nodes {num_nodes} "
                f"(got {self.weight.numel()})")
        return torch.multinomial(weight, num_samples, replacement=True)


class BaseSampler(ABC):
    r"""An abstract base class that initializes a graph sampler and provides
    :meth:`sample_from_nodes` and :meth:`sample_from_edges` routines.

    .. note ::

        Any data stored in the sampler will be *replicated* across data loading
        workers that use the sampler since each data loading worker holds its
        own instance of a sampler.
        As such, it is recommended to limit the amount of information stored in
        the sampler.
    """
    @abstractmethod
    def sample_from_nodes(
        self,
        index: NodeSamplerInput,
        **kwargs,
    ) -> Union[HeteroSamplerOutput, SamplerOutput]:
        r"""Performs sampling from the nodes specified in :obj:`index`,
        returning a sampled subgraph in the specified output format.

        The :obj:`index` is a tuple holding the following information:

        1. The example indices of the seed nodes
        2. The node indices to start sampling from
        3. The timestamps of the given seed nodes (optional)

        Args:
            index (NodeSamplerInput): The node sampler input object.
            **kwargs (optional): Additional keyword arguments.
        """
        raise NotImplementedError

    @abstractmethod
    def sample_from_edges(
        self,
        index: EdgeSamplerInput,
        neg_sampling: Optional[NegativeSampling] = None,
    ) -> Union[HeteroSamplerOutput, SamplerOutput]:
        r"""Performs sampling from the edges specified in :obj:`index`,
        returning a sampled subgraph in the specified output format.

        The :obj:`index` is a tuple holding the following information:

        1. The example indices of the seed links
        2. The source node indices to start sampling from
        3. The destination node indices to start sampling from
        4. The labels of the seed links (optional)
        5. The timestamps of the given seed nodes (optional)

        Args:
            index (EdgeSamplerInput): The edge sampler input object.
            neg_sampling (NegativeSampling, optional): The negative sampling
                configuration. (default: :obj:`None`)
        """
        raise NotImplementedError

    @property
    def edge_permutation(self) -> Union[OptTensor, Dict[EdgeType, OptTensor]]:
        r"""If the sampler performs any modification of edge ordering in the
        original graph, this function is expected to return the permutation
        tensor that defines the permutation from the edges in the original
        graph and the edges used in the sampler. If no such permutation was
        applied, :obj:`None` is returned. For heterogeneous graphs, the
        expected return type is a permutation tensor for each edge type.
        """
        return None
