import copy
import math
import sys
import warnings
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union

import torch
from torch import Tensor

import torch_geometric.typing
from torch_geometric.data import (
    Data,
    FeatureStore,
    GraphStore,
    HeteroData,
    remote_backend_utils,
)
from torch_geometric.data.graph_store import EdgeLayout
from torch_geometric.sampler import (
    BaseSampler,
    EdgeSamplerInput,
    HeteroSamplerOutput,
    NegativeSampling,
    NodeSamplerInput,
    SamplerOutput,
)
from torch_geometric.sampler.base import DataType, NumNeighbors, SubgraphType
from torch_geometric.sampler.utils import remap_keys, to_csc, to_hetero_csc
from torch_geometric.typing import EdgeType, NodeType, OptTensor

NumNeighborsType = Union[NumNeighbors, List[int], Dict[EdgeType, List[int]]]


class NeighborSampler(BaseSampler):
    r"""An implementation of an in-memory (heterogeneous) neighbor sampler used
    by :class:`~torch_geometric.loader.NeighborLoader`.
    """
    def __init__(
        self,
        data: Union[Data, HeteroData, Tuple[FeatureStore, GraphStore]],
        num_neighbors: NumNeighborsType,
        subgraph_type: Union[SubgraphType, str] = 'directional',
        replace: bool = False,
        disjoint: bool = False,
        temporal_strategy: str = 'uniform',
        time_attr: Optional[str] = None,
        weight_attr: Optional[str] = None,
        is_sorted: bool = False,
        share_memory: bool = False,
        # Deprecated:
        directed: bool = True,
    ):
        if not directed:
            subgraph_type = SubgraphType.induced
            warnings.warn(f"The usage of the 'directed' argument in "
                          f"'{self.__class__.__name__}' is deprecated. Use "
                          f"`subgraph_type='induced'` instead.")

        if (not torch_geometric.typing.WITH_PYG_LIB and sys.platform == 'linux'
                and subgraph_type != SubgraphType.induced):
            warnings.warn(f"Using '{self.__class__.__name__}' without a "
                          f"'pyg-lib' installation is deprecated and will be "
                          f"removed soon. Please install 'pyg-lib' for "
                          f"accelerated neighborhood sampling")

        self.data_type = DataType.from_data(data)

        if self.data_type == DataType.homogeneous:
            self.num_nodes = data.num_nodes

            self.node_time: Optional[Tensor] = None
            self.edge_time: Optional[Tensor] = None

            if time_attr is not None:
                if data.is_node_attr(time_attr):
                    self.node_time = data[time_attr]
                elif data.is_edge_attr(time_attr):
                    self.edge_time = data[time_attr]
                else:
                    raise ValueError(
                        f"The time attribute '{time_attr}' is neither a "
                        f"node-level or edge-level attribute")

            # Convert the graph data into CSC format for sampling:
            self.colptr, self.row, self.perm = to_csc(
                data, device='cpu', share_memory=share_memory,
                is_sorted=is_sorted, src_node_time=self.node_time,
                edge_time=self.edge_time)

            if self.edge_time is not None and self.perm is not None:
                self.edge_time = self.edge_time[self.perm]

            self.edge_weight: Optional[Tensor] = None
            if weight_attr is not None:
                self.edge_weight = data[weight_attr]
                if self.perm is not None:
                    self.edge_weight = self.edge_weight[self.perm]

        elif self.data_type == DataType.heterogeneous:
            self.node_types, self.edge_types = data.metadata()

            self.num_nodes = {k: data[k].num_nodes for k in self.node_types}

            self.node_time: Optional[Dict[NodeType, Tensor]] = None
            self.edge_time: Optional[Dict[EdgeType, Tensor]] = None

            if time_attr is not None:
                is_node_level_time = is_edge_level_time = False

                for store in data.node_stores:
                    if time_attr in store:
                        is_node_level_time = True
                for store in data.edge_stores:
                    if time_attr in store:
                        is_edge_level_time = True

                if is_node_level_time and is_edge_level_time:
                    raise ValueError(
                        f"The time attribute '{time_attr}' holds both "
                        f"node-level and edge-level information")

                if not is_node_level_time and not is_edge_level_time:
                    raise ValueError(
                        f"The time attribute '{time_attr}' is neither a "
                        f"node-level or edge-level attribute")

                if is_node_level_time:
                    self.node_time = data.collect(time_attr)
                else:
                    self.edge_time = data.collect(time_attr)

            # Conversion to/from C++ string type: Since C++ cannot take
            # dictionaries with tuples as key as input, edge type triplets need
            # to be converted into single strings.
            self.to_rel_type = {k: '__'.join(k) for k in self.edge_types}
            self.to_edge_type = {v: k for k, v in self.to_rel_type.items()}

            # Convert the graph data into CSC format for sampling:
            colptr_dict, row_dict, self.perm = to_hetero_csc(
                data, device='cpu', share_memory=share_memory,
                is_sorted=is_sorted, node_time_dict=self.node_time,
                edge_time_dict=self.edge_time)

            self.row_dict = remap_keys(row_dict, self.to_rel_type)
            self.colptr_dict = remap_keys(colptr_dict, self.to_rel_type)

            if self.edge_time is not None:
                for edge_type, edge_time in self.edge_time.items():
                    if self.perm.get(edge_type, None) is not None:
                        edge_time = edge_time[self.perm[edge_type]]
                        self.edge_time[edge_type] = edge_time
                self.edge_time = remap_keys(self.edge_time, self.to_rel_type)

            self.edge_weight: Optional[Dict[EdgeType, Tensor]] = None
            if weight_attr is not None:
                self.edge_weight = data.collect(weight_attr)
                for edge_type, edge_weight in self.edge_weight.items():
                    if self.perm.get(edge_type, None) is not None:
                        edge_weight = edge_weight[self.perm[edge_type]]
                        self.edge_weight[edge_type] = edge_weight
                self.edge_weight = remap_keys(self.edge_weight,
                                              self.to_rel_type)

        else:  # self.data_type == DataType.remote
            feature_store, graph_store = data

            # Obtain graph metadata:
            attrs = [attr for attr in feature_store.get_all_tensor_attrs()]

            edge_attrs = graph_store.get_all_edge_attrs()
            self.edge_types = list({attr.edge_type for attr in edge_attrs})

            if weight_attr is not None:
                raise NotImplementedError(
                    f"'weight_attr' argument not yet supported within "
                    f"'{self.__class__.__name__}' for "
                    f"'(FeatureStore, GraphStore)' inputs")

            if time_attr is not None:
                # If the `time_attr` is present, we expect that `GraphStore`
                # holds all edges sorted by destination, and within local
                # neighborhoods, node indices should be sorted by time.
                # TODO (matthias, manan) Find an alternative way to ensure.
                for edge_attr in edge_attrs:
                    if edge_attr.layout == EdgeLayout.CSR:
                        raise ValueError(
                            "Temporal sampling requires that edges are stored "
                            "in either COO or CSC layout")
                    if not edge_attr.is_sorted:
                        raise ValueError(
                            "Temporal sampling requires that edges are "
                            "sorted by destination, and by source time "
                            "within local neighborhoods")

                # We obtain all features with `node_attr.name=time_attr`:
                time_attrs = [
                    copy.copy(attr) for attr in attrs
                    if attr.attr_name == time_attr
                ]

            if not self.is_hetero:
                self.node_types = [None]
                self.num_nodes = max(edge_attrs[0].size)
                self.edge_weight: Optional[Tensor] = None

                self.node_time: Optional[Tensor] = None
                self.edge_time: Optional[Tensor] = None

                if time_attr is not None:
                    if len(time_attrs) != 1:
                        raise ValueError("Temporal sampling specified but did "
                                         "not find any temporal data")
                    time_attrs[0].index = None  # Reset index for full data.
                    time_tensor = feature_store.get_tensor(time_attrs[0])
                    # Currently, we determine whether to use node-level or
                    # edge-level temporal sampling based on the attribute name.
                    if time_attr == 'time':
                        self.node_time = time_tensor
                    else:
                        self.edge_time = time_tensor

                self.row, self.colptr, self.perm = graph_store.csc()

            else:
                node_types = [
                    attr.group_name for attr in attrs
                    if isinstance(attr.group_name, str)
                ]
                self.node_types = list(set(node_types))
                self.num_nodes = {
                    node_type: remote_backend_utils.size(*data, node_type)
                    for node_type in self.node_types
                }
                self.edge_weight: Optional[Dict[EdgeType, Tensor]] = None

                self.node_time: Optional[Dict[NodeType, Tensor]] = None
                self.edge_time: Optional[Dict[EdgeType, Tensor]] = None

                if time_attr is not None:
                    for attr in time_attrs:  # Reset index for full data.
                        attr.index = None

                    time_tensors = feature_store.multi_get_tensor(time_attrs)
                    time = {
                        attr.group_name: time_tensor
                        for attr, time_tensor in zip(time_attrs, time_tensors)
                    }

                    group_names = [attr.group_name for attr in time_attrs]
                    if all([isinstance(g, str) for g in group_names]):
                        self.node_time = time
                    elif all([isinstance(g, tuple) for g in group_names]):
                        self.edge_time = time
                    else:
                        raise ValueError(
                            f"Found time attribute '{time_attr}' for both "
                            f"node-level and edge-level types")

                # Conversion to/from C++ string type (see above):
                self.to_rel_type = {k: '__'.join(k) for k in self.edge_types}
                self.to_edge_type = {v: k for k, v in self.to_rel_type.items()}
                # Convert the graph data into CSC format for sampling:
                row_dict, colptr_dict, self.perm = graph_store.csc()
                self.row_dict = remap_keys(row_dict, self.to_rel_type)
                self.colptr_dict = remap_keys(colptr_dict, self.to_rel_type)

        if (self.edge_time is not None
                and not torch_geometric.typing.WITH_EDGE_TIME_NEIGHBOR_SAMPLE):
            raise ImportError("Edge-level temporal sampling requires a "
                              "more recent 'pyg-lib' installation")

        if (self.edge_weight is not None
                and not torch_geometric.typing.WITH_WEIGHTED_NEIGHBOR_SAMPLE):
            raise ImportError("Weighted neighbor sampling requires "
                              "'pyg-lib>=0.3.0'")

        self.num_neighbors = num_neighbors
        self.replace = replace
        self.subgraph_type = SubgraphType(subgraph_type)
        self.disjoint = disjoint
        self.temporal_strategy = temporal_strategy
        self.keep_orig_edges = False

    @property
    def num_neighbors(self) -> NumNeighbors:
        return self._num_neighbors

    @num_neighbors.setter
    def num_neighbors(self, num_neighbors: NumNeighborsType):
        if isinstance(num_neighbors, NumNeighbors):
            self._num_neighbors = num_neighbors
        else:
            self._num_neighbors = NumNeighbors(num_neighbors)

    @property
    def is_hetero(self) -> bool:
        if self.data_type == DataType.homogeneous:
            return False
        if self.data_type == DataType.heterogeneous:
            return True

        # self.data_type == DataType.remote
        return self.edge_types != [None]

    @property
    def is_temporal(self) -> bool:
        return self.node_time is not None or self.edge_time is not None

    @property
    def disjoint(self) -> bool:
        return self._disjoint or self.is_temporal

    @disjoint.setter
    def disjoint(self, disjoint: bool):
        self._disjoint = disjoint

    # Node-based sampling #####################################################

    def sample_from_nodes(
        self,
        inputs: NodeSamplerInput,
    ) -> Union[SamplerOutput, HeteroSamplerOutput]:
        out = node_sample(inputs, self._sample)
        if self.subgraph_type == SubgraphType.bidirectional:
            out = out.to_bidirectional(keep_orig_edges=self.keep_orig_edges)
        return out

    # Edge-based sampling #####################################################

    def sample_from_edges(
        self,
        inputs: EdgeSamplerInput,
        neg_sampling: Optional[NegativeSampling] = None,
    ) -> Union[SamplerOutput, HeteroSamplerOutput]:
        out = edge_sample(inputs, self._sample, self.num_nodes, self.disjoint,
                          self.node_time, neg_sampling)
        if self.subgraph_type == SubgraphType.bidirectional:
            out = out.to_bidirectional(keep_orig_edges=self.keep_orig_edges)
        return out

    # Other Utilities #########################################################

    @property
    def edge_permutation(self) -> Union[OptTensor, Dict[EdgeType, OptTensor]]:
        return self.perm

    # Helper functions ########################################################

    def _sample(
        self,
        seed: Union[Tensor, Dict[NodeType, Tensor]],
        seed_time: Optional[Union[Tensor, Dict[NodeType, Tensor]]] = None,
        **kwargs,
    ) -> Union[SamplerOutput, HeteroSamplerOutput]:
        r"""Implements neighbor sampling by calling either :obj:`pyg-lib` (if
        installed) or :obj:`torch-sparse` (if installed) sampling routines.
        """
        if isinstance(seed, dict):  # Heterogeneous sampling:
            # TODO Support induced subgraph sampling in `pyg-lib`.
            if (torch_geometric.typing.WITH_PYG_LIB
                    and self.subgraph_type != SubgraphType.induced):
                # TODO (matthias) Ideally, `seed` inherits dtype from `colptr`
                colptrs = list(self.colptr_dict.values())
                dtype = colptrs[0].dtype if len(colptrs) > 0 else torch.int64
                seed = {k: v.to(dtype) for k, v in seed.items()}

                args = (
                    self.node_types,
                    self.edge_types,
                    self.colptr_dict,
                    self.row_dict,
                    seed,
                    self.num_neighbors.get_mapped_values(self.edge_types),
                    self.node_time,
                )
                if torch_geometric.typing.WITH_EDGE_TIME_NEIGHBOR_SAMPLE:
                    args += (self.edge_time, )
                args += (seed_time, )
                if torch_geometric.typing.WITH_WEIGHTED_NEIGHBOR_SAMPLE:
                    args += (self.edge_weight, )
                args += (
                    True,  # csc
                    self.replace,
                    self.subgraph_type != SubgraphType.induced,
                    self.disjoint,
                    self.temporal_strategy,
                    # TODO (matthias) `return_edge_id` if edge features present
                    True,  # return_edge_id
                )

                out = torch.ops.pyg.hetero_neighbor_sample(*args)
                row, col, node, edge, batch = out[:4] + (None, )

                # `pyg-lib>0.1.0` returns sampled number of nodes/edges:
                num_sampled_nodes = num_sampled_edges = None
                if len(out) >= 6:
                    num_sampled_nodes, num_sampled_edges = out[4:6]

                if self.disjoint:
                    node = {k: v.t().contiguous() for k, v in node.items()}
                    batch = {k: v[0] for k, v in node.items()}
                    node = {k: v[1] for k, v in node.items()}

            elif torch_geometric.typing.WITH_TORCH_SPARSE:
                if self.disjoint:
                    if self.subgraph_type == SubgraphType.induced:
                        raise ValueError("'disjoint' sampling not supported "
                                         "for neighbor sampling with "
                                         "`subgraph_type='induced'`")
                    else:
                        raise ValueError("'disjoint' sampling not supported "
                                         "for neighbor sampling via "
                                         "'torch-sparse'. Please install "
                                         "'pyg-lib' for improved and "
                                         "optimized sampling routines.")

                out = torch.ops.torch_sparse.hetero_neighbor_sample(
                    self.node_types,
                    self.edge_types,
                    self.colptr_dict,
                    self.row_dict,
                    seed,  # seed_dict
                    self.num_neighbors.get_mapped_values(self.edge_types),
                    self.num_neighbors.num_hops,
                    self.replace,
                    self.subgraph_type != SubgraphType.induced,
                )
                node, row, col, edge, batch = out + (None, )
                num_sampled_nodes = num_sampled_edges = None

            else:
                raise ImportError(f"'{self.__class__.__name__}' requires "
                                  f"either 'pyg-lib' or 'torch-sparse'")

            if num_sampled_edges is not None:
                num_sampled_edges = remap_keys(
                    num_sampled_edges,
                    self.to_edge_type,
                )

            return HeteroSamplerOutput(
                node=node,
                row=remap_keys(row, self.to_edge_type),
                col=remap_keys(col, self.to_edge_type),
                edge=remap_keys(edge, self.to_edge_type),
                batch=batch,
                num_sampled_nodes=num_sampled_nodes,
                num_sampled_edges=num_sampled_edges,
            )

        else:  # Homogeneous sampling:
            # TODO Support induced subgraph sampling in `pyg-lib`.
            if (torch_geometric.typing.WITH_PYG_LIB
                    and self.subgraph_type != SubgraphType.induced):

                args = (
                    self.colptr,
                    self.row,
                    # TODO (matthias) `seed` should inherit dtype from `colptr`
                    seed.to(self.colptr.dtype),
                    self.num_neighbors.get_mapped_values(),
                    self.node_time,
                )
                if torch_geometric.typing.WITH_EDGE_TIME_NEIGHBOR_SAMPLE:
                    args += (self.edge_time, )
                args += (seed_time, )
                if torch_geometric.typing.WITH_WEIGHTED_NEIGHBOR_SAMPLE:
                    args += (self.edge_weight, )
                args += (
                    True,  # csc
                    self.replace,
                    self.subgraph_type != SubgraphType.induced,
                    self.disjoint,
                    self.temporal_strategy,
                    # TODO (matthias) `return_edge_id` if edge features present
                    True,  # return_edge_id
                )

                out = torch.ops.pyg.neighbor_sample(*args)
                row, col, node, edge, batch = out[:4] + (None, )

                # `pyg-lib>0.1.0` returns sampled number of nodes/edges:
                num_sampled_nodes = num_sampled_edges = None
                if len(out) >= 6:
                    num_sampled_nodes, num_sampled_edges = out[4:6]

                if self.disjoint:
                    batch, node = node.t().contiguous()

            elif torch_geometric.typing.WITH_TORCH_SPARSE:
                if self.disjoint:
                    raise ValueError("'disjoint' sampling not supported for "
                                     "neighbor sampling via 'torch-sparse'. "
                                     "Please install 'pyg-lib' for improved "
                                     "and optimized sampling routines.")

                out = torch.ops.torch_sparse.neighbor_sample(
                    self.colptr,
                    self.row,
                    seed,  # seed
                    self.num_neighbors.get_mapped_values(),
                    self.replace,
                    self.subgraph_type != SubgraphType.induced,
                )
                node, row, col, edge, batch = out + (None, )
                num_sampled_nodes = num_sampled_edges = None

            else:
                raise ImportError(f"'{self.__class__.__name__}' requires "
                                  f"either 'pyg-lib' or 'torch-sparse'")

            return SamplerOutput(
                node=node,
                row=row,
                col=col,
                edge=edge,
                batch=batch,
                num_sampled_nodes=num_sampled_nodes,
                num_sampled_edges=num_sampled_edges,
            )


# Sampling Utilities ##########################################################


def node_sample(
    inputs: NodeSamplerInput,
    sample_fn: Callable,
) -> Union[SamplerOutput, HeteroSamplerOutput]:
    r"""Performs sampling from a :class:`NodeSamplerInput`, leveraging a
    sampling function that accepts a seed and (optionally) a seed time as
    input. Returns the output of this sampling procedure.
    """
    if inputs.input_type is not None:  # Heterogeneous sampling:
        seed = {inputs.input_type: inputs.node}
        seed_time = None
        if inputs.time is not None:
            seed_time = {inputs.input_type: inputs.time}
    else:  # Homogeneous sampling:
        seed = inputs.node
        seed_time = inputs.time

    out = sample_fn(seed, seed_time)
    out.metadata = (inputs.input_id, inputs.time)

    return out


def edge_sample(
    inputs: EdgeSamplerInput,
    sample_fn: Callable,
    num_nodes: Union[int, Dict[NodeType, int]],
    disjoint: bool,
    node_time: Optional[Union[Tensor, Dict[str, Tensor]]] = None,
    neg_sampling: Optional[NegativeSampling] = None,
) -> Union[SamplerOutput, HeteroSamplerOutput]:
    r"""Performs sampling from an edge sampler input, leveraging a sampling
    function of the same signature as `node_sample`.
    """
    input_id = inputs.input_id
    src = inputs.row
    dst = inputs.col
    edge_label = inputs.label
    edge_label_time = inputs.time
    input_type = inputs.input_type

    src_time = dst_time = edge_label_time
    assert edge_label_time is None or disjoint

    assert isinstance(num_nodes, (dict, int))
    if not isinstance(num_nodes, dict):
        num_src_nodes = num_dst_nodes = num_nodes
    else:
        num_src_nodes = num_nodes[input_type[0]]
        num_dst_nodes = num_nodes[input_type[-1]]

    num_pos = src.numel()
    num_neg = 0

    # Negative Sampling #######################################################

    if neg_sampling is not None:
        # When we are doing negative sampling, we append negative information
        # of nodes/edges to `src`, `dst`, `src_time`, `dst_time`.
        # Later on, we can easily reconstruct what belongs to positive and
        # negative examples by slicing via `num_pos`.
        num_neg = math.ceil(num_pos * neg_sampling.amount)

        if neg_sampling.is_binary():
            # In the "binary" case, we randomly sample negative pairs of nodes.
            if isinstance(node_time, dict):
                src_node_time = node_time.get(input_type[0])
            else:
                src_node_time = node_time

            src_neg = neg_sample(src, neg_sampling, num_src_nodes, src_time,
                                 src_node_time, endpoint='src')
            src = torch.cat([src, src_neg], dim=0)

            if isinstance(node_time, dict):
                dst_node_time = node_time.get(input_type[-1])
            else:
                dst_node_time = node_time

            dst_neg = neg_sample(dst, neg_sampling, num_dst_nodes, dst_time,
                                 dst_node_time, endpoint='dst')
            dst = torch.cat([dst, dst_neg], dim=0)

            if edge_label is None:
                edge_label = torch.ones(num_pos)
            size = (num_neg, ) + edge_label.size()[1:]
            edge_neg_label = edge_label.new_zeros(size)
            edge_label = torch.cat([edge_label, edge_neg_label])

            if edge_label_time is not None:
                src_time = dst_time = edge_label_time.repeat(
                    1 + math.ceil(neg_sampling.amount))[:num_pos + num_neg]

        elif neg_sampling.is_triplet():
            # In the "triplet" case, we randomly sample negative destinations.
            if isinstance(node_time, dict):
                dst_node_time = node_time.get(input_type[-1])
            else:
                dst_node_time = node_time

            dst_neg = neg_sample(dst, neg_sampling, num_dst_nodes, dst_time,
                                 dst_node_time, endpoint='dst')
            dst = torch.cat([dst, dst_neg], dim=0)

            assert edge_label is None

            if edge_label_time is not None:
                dst_time = edge_label_time.repeat(1 + neg_sampling.amount)

    # Heterogeneous Neighborhood Sampling #####################################

    if input_type is not None:
        seed_time_dict = None
        if input_type[0] != input_type[-1]:  # Two distinct node types:

            if not disjoint:
                src, inverse_src = src.unique(return_inverse=True)
                dst, inverse_dst = dst.unique(return_inverse=True)

            seed_dict = {input_type[0]: src, input_type[-1]: dst}

            if edge_label_time is not None:  # Always disjoint.
                seed_time_dict = {
                    input_type[0]: src_time,
                    input_type[-1]: dst_time,
                }

        else:  # Only a single node type: Merge both source and destination.

            seed = torch.cat([src, dst], dim=0)

            if not disjoint:
                seed, inverse_seed = seed.unique(return_inverse=True)

            seed_dict = {input_type[0]: seed}

            if edge_label_time is not None:  # Always disjoint.
                seed_time_dict = {
                    input_type[0]: torch.cat([src_time, dst_time], dim=0),
                }

        out = sample_fn(seed_dict, seed_time_dict)

        # Enhance `out` by label information ##################################
        if disjoint:
            for key, batch in out.batch.items():
                out.batch[key] = batch % num_pos

        if neg_sampling is None or neg_sampling.is_binary():
            if disjoint:
                if input_type[0] != input_type[-1]:
                    edge_label_index = torch.arange(num_pos + num_neg)
                    edge_label_index = edge_label_index.repeat(2).view(2, -1)
                else:
                    edge_label_index = torch.arange(2 * (num_pos + num_neg))
                    edge_label_index = edge_label_index.view(2, -1)
            else:
                if input_type[0] != input_type[-1]:
                    edge_label_index = torch.stack([
                        inverse_src,
                        inverse_dst,
                    ], dim=0)
                else:
                    edge_label_index = inverse_seed.view(2, -1)

            out.metadata = (input_id, edge_label_index, edge_label, src_time)

        elif neg_sampling.is_triplet():
            if disjoint:
                src_index = torch.arange(num_pos)
                if input_type[0] != input_type[-1]:
                    dst_pos_index = torch.arange(num_pos)
                    # `dst_neg_index` needs to be offset such that indices with
                    # offset `num_pos` belong to the same triplet:
                    dst_neg_index = torch.arange(
                        num_pos, seed_dict[input_type[-1]].numel())
                    dst_neg_index = dst_neg_index.view(-1, num_pos).t()
                else:
                    dst_pos_index = torch.arange(num_pos, 2 * num_pos)
                    dst_neg_index = torch.arange(
                        2 * num_pos, seed_dict[input_type[-1]].numel())
                    dst_neg_index = dst_neg_index.view(-1, num_pos).t()
            else:
                if input_type[0] != input_type[-1]:
                    src_index = inverse_src
                    dst_pos_index = inverse_dst[:num_pos]
                    dst_neg_index = inverse_dst[num_pos:]
                else:
                    src_index = inverse_seed[:num_pos]
                    dst_pos_index = inverse_seed[num_pos:2 * num_pos]
                    dst_neg_index = inverse_seed[2 * num_pos:]

            dst_neg_index = dst_neg_index.view(num_pos, -1).squeeze(-1)

            out.metadata = (
                input_id,
                src_index,
                dst_pos_index,
                dst_neg_index,
                src_time,
            )

    # Homogeneous Neighborhood Sampling #######################################

    else:

        seed = torch.cat([src, dst], dim=0)
        seed_time = None

        if not disjoint:
            seed, inverse_seed = seed.unique(return_inverse=True)

        if edge_label_time is not None:  # Always disjoint.
            seed_time = torch.cat([src_time, dst_time])

        out = sample_fn(seed, seed_time)

        # Enhance `out` by label information ##################################
        if neg_sampling is None or neg_sampling.is_binary():
            if disjoint:
                out.batch = out.batch % num_pos
                edge_label_index = torch.arange(seed.numel()).view(2, -1)
            else:
                edge_label_index = inverse_seed.view(2, -1)

            out.metadata = (input_id, edge_label_index, edge_label, src_time)

        elif neg_sampling.is_triplet():
            if disjoint:
                out.batch = out.batch % num_pos
                src_index = torch.arange(num_pos)
                dst_pos_index = torch.arange(num_pos, 2 * num_pos)
                # `dst_neg_index` needs to be offset such that indices with
                # offset `num_pos` belong to the same triplet:
                dst_neg_index = torch.arange(2 * num_pos, seed.numel())
                dst_neg_index = dst_neg_index.view(-1, num_pos).t()
            else:
                src_index = inverse_seed[:num_pos]
                dst_pos_index = inverse_seed[num_pos:2 * num_pos]
                dst_neg_index = inverse_seed[2 * num_pos:]
            dst_neg_index = dst_neg_index.view(num_pos, -1).squeeze(-1)

            out.metadata = (
                input_id,
                src_index,
                dst_pos_index,
                dst_neg_index,
                src_time,
            )

    return out


def neg_sample(
    seed: Tensor,
    neg_sampling: NegativeSampling,
    num_nodes: int,
    seed_time: Optional[Tensor],
    node_time: Optional[Tensor],
    endpoint: Literal['str', 'dst'],
) -> Tensor:
    num_neg = math.ceil(seed.numel() * neg_sampling.amount)

    # TODO: Do not sample false negatives.
    if node_time is None:
        return neg_sampling.sample(num_neg, endpoint, num_nodes)

    # If we are in a temporal-sampling scenario, we need to respect the
    # timestamp of the given nodes we can use as negative examples.
    # That is, we can only sample nodes for which `node_time <= seed_time`.
    # For now, we use a greedy algorithm which randomly samples negative
    # nodes and discard any which do not respect the temporal constraint.
    # We iteratively repeat this process until we have sampled a valid node for
    # each seed.
    # TODO See if this greedy algorithm here can be improved.
    assert seed_time is not None
    num_samples = math.ceil(neg_sampling.amount)
    seed_time = seed_time.view(1, -1).expand(num_samples, -1)

    out = neg_sampling.sample(num_samples * seed.numel(), endpoint, num_nodes)
    out = out.view(num_samples, seed.numel())
    mask = node_time[out] > seed_time  # holds all invalid samples.
    neg_sampling_complete = False
    for i in range(5):  # pragma: no cover
        num_invalid = int(mask.sum())
        if num_invalid == 0:
            neg_sampling_complete = True
            break

        # Greedily search for alternative negatives.
        out[mask] = tmp = neg_sampling.sample(num_invalid, endpoint, num_nodes)
        mask[mask.clone()] = node_time[tmp] >= seed_time[mask]

    if not neg_sampling_complete:  # pragma: no cover
        # Not much options left. In that case, we set remaining negatives
        # to the node with minimum timestamp.
        out[mask] = node_time.argmin()

    return out.view(-1)[:num_neg]
