from typing import Any, Dict, Optional, Tuple, Union

import torch

from torch_geometric.data import Data, HeteroData, remote_backend_utils
from torch_geometric.data.feature_store import FeatureStore
from torch_geometric.data.graph_store import EdgeLayout, GraphStore
from torch_geometric.sampler.base import (
    BaseSampler,
    EdgeSamplerInput,
    HeteroSamplerOutput,
    NodeSamplerInput,
    SamplerOutput,
)
from torch_geometric.sampler.utils import (
    add_negative_samples,
    remap_keys,
    set_node_time_dict,
    to_csc,
    to_hetero_csc,
)
from torch_geometric.typing import EdgeType, NodeType, NumNeighbors, OptTensor

try:
    import pyg_lib  # noqa
    _WITH_PYG_LIB = True
except ImportError:
    _WITH_PYG_LIB = False


class NeighborSampler(BaseSampler):
    r"""An implementation of an in-memory (heterogeneous) neighbor sampler used
    by :class:`~torch_geometric.loader.NeighborLoader`."""
    def __init__(
        self,
        data: Union[Data, HeteroData, Tuple[FeatureStore, GraphStore]],
        num_neighbors: NumNeighbors,
        replace: bool = False,
        directed: bool = True,
        input_type: Optional[Any] = None,
        time_attr: Optional[str] = None,
        is_sorted: bool = False,
        share_memory: bool = False,
    ):
        self.data_cls = data.__class__ if isinstance(
            data, (Data, HeteroData)) else 'custom'
        self.num_neighbors = num_neighbors
        self.replace = replace
        self.directed = directed
        self.node_time = None
        self.input_type = input_type

        # TODO if self.edge_time is not None and
        # `src` or `dst` nodes don't have time attribute
        # i.e node_time_dict[input_type[0/-1]] doesn't exist
        # set it to largest representable torch.long.

        # Set the number of source and destination nodes if we can, otherwise
        # ignore:
        self.num_src_nodes, self.num_dst_nodes = None, None
        if self.data_cls != 'custom' and issubclass(self.data_cls, Data):
            self.num_src_nodes = self.num_dst_nodes = data.num_nodes
        elif isinstance(self.input_type, tuple):
            if self.data_cls == 'custom':
                out = remote_backend_utils.size(*data, self.input_type)
                self.num_src_nodes, self.num_dst_nodes = out
            else:  # issubclass(self.data_cls, HeteroData):
                self.num_src_nodes = data[self.input_type[0]].num_nodes
                self.num_dst_nodes = data[self.input_type[-1]].num_nodes

        # TODO Unify the following conditionals behind the `FeatureStore`
        # and `GraphStore` API:

        # If we are working with a `Data` object, convert the edge_index to
        # CSC and store it:
        if isinstance(data, Data):
            self.node_time = None
            if time_attr is not None:
                self.node_time = data[time_attr]

            # Convert the graph data into a suitable format for sampling.
            out = to_csc(data, device='cpu', share_memory=share_memory,
                         is_sorted=is_sorted, src_node_time=self.node_time)
            self.colptr, self.row, self.perm = out
            assert isinstance(num_neighbors, (list, tuple))

        # If we are working with a `HeteroData` object, convert each edge
        # type's edge_index to CSC and store it:
        elif isinstance(data, HeteroData):
            self.node_time_dict = None
            if time_attr is not None:
                self.node_time_dict = data.collect(time_attr)

            self.node_types, self.edge_types = data.metadata()
            self._set_num_neighbors_and_num_hops(num_neighbors)

            assert input_type is not None
            self.input_type = input_type

            # Obtain CSC representations for in-memory sampling:
            out = to_hetero_csc(data, device='cpu', share_memory=share_memory,
                                is_sorted=is_sorted,
                                node_time_dict=self.node_time_dict)
            colptr_dict, row_dict, perm_dict = out

            # Conversions to/from C++ string type:
            # Since C++ cannot take dictionaries with tuples as key as input,
            # edge type triplets need to be converted into single strings. This
            # is done by maintaining the following mappings:
            self.to_rel_type = {key: '__'.join(key) for key in self.edge_types}
            self.to_edge_type = {
                '__'.join(key): key
                for key in self.edge_types
            }

            self.row_dict = remap_keys(row_dict, self.to_rel_type)
            self.colptr_dict = remap_keys(colptr_dict, self.to_rel_type)
            self.num_neighbors = remap_keys(self.num_neighbors,
                                            self.to_rel_type)
            self.perm = perm_dict

        # If we are working with a `Tuple[FeatureStore, GraphStore]` object,
        # obtain edges from GraphStore and convert them to CSC if necessary,
        # storing the resulting representations:
        elif isinstance(data, tuple):
            # TODO support `FeatureStore` with no edge types (e.g. `Data`)
            feature_store, graph_store = data

            # Obtain all node and edge metadata:
            node_attrs = feature_store.get_all_tensor_attrs()
            edge_attrs = graph_store.get_all_edge_attrs()

            # TODO support `collect` on `FeatureStore`:
            self.node_time_dict = None
            if time_attr is not None:
                # If the `time_attr` is present, we expect that `GraphStore`
                # holds all edges sorted by destination, and within local
                # neighborhoods, node indices should be sorted by time.
                # TODO (matthias, manan) Find an alternative way to ensure
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

                # We need to obtain all features with 'attr_name=time_attr'
                # from the feature store and store them in node_time_dict. To
                # do so, we make an explicit feature store GET call here with
                # the relevant 'TensorAttr's
                time_attrs = [
                    attr for attr in node_attrs if attr.attr_name == time_attr
                ]
                for attr in time_attrs:
                    attr.index = None
                time_tensors = feature_store.multi_get_tensor(time_attrs)
                self.node_time_dict = {
                    time_attr.group_name: time_tensor
                    for time_attr, time_tensor in zip(time_attrs, time_tensors)
                }

            self.node_types = list(
                set(node_attr.group_name for node_attr in node_attrs))
            self.edge_types = list(
                set(edge_attr.edge_type for edge_attr in edge_attrs))

            self._set_num_neighbors_and_num_hops(num_neighbors)

            assert input_type is not None
            self.input_type = input_type

            # Obtain CSC representations for in-memory sampling:
            row_dict, colptr_dict, perm_dict = graph_store.csc()

            self.to_rel_type = {key: '__'.join(key) for key in self.edge_types}
            self.to_edge_type = {
                '__'.join(key): key
                for key in self.edge_types
            }
            self.row_dict = remap_keys(row_dict, self.to_rel_type)
            self.colptr_dict = remap_keys(colptr_dict, self.to_rel_type)
            self.num_neighbors = remap_keys(self.num_neighbors,
                                            self.to_rel_type)
            self.perm = perm_dict

        else:
            raise TypeError(f"'{self.__class__.__name__}'' found invalid "
                            f"type: '{type(data)}'")

    def _set_num_neighbors_and_num_hops(self, num_neighbors):
        if isinstance(num_neighbors, (list, tuple)):
            num_neighbors = {key: num_neighbors for key in self.edge_types}
        assert isinstance(num_neighbors, dict)
        self.num_neighbors = num_neighbors

        # Add at least one element to the list to ensure `max` is well-defined
        self.num_hops = max([0] + [len(v) for v in num_neighbors.values()])

        for key, value in self.num_neighbors.items():
            if len(value) != self.num_hops:
                raise ValueError(f"Expected the edge type {key} to have "
                                 f"{self.num_hops} entries (got {len(value)})")

    def _sample(
        self,
        seed: Union[torch.Tensor, Dict[NodeType, torch.Tensor]],
        **kwargs,
    ) -> Union[SamplerOutput, HeteroSamplerOutput]:
        r"""Implements neighbor sampling by calling :obj:`pyg-lib` or
        :obj:`torch-sparse` sampling routines, conditional on the type of
        :obj:`data` object.

        Note that the 'metadata' field of the output is not filled; it is the
        job of the caller to appropriately fill out this field for downstream
        loaders."""

        # TODO(manan): remote backends only support heterogeneous graphs for
        # now:
        if self.data_cls == 'custom' or issubclass(self.data_cls, HeteroData):
            if _WITH_PYG_LIB:
                # TODO (matthias) Add `disjoint` option to `NeighborSampler`
                # TODO (matthias) `return_edge_id` if edge features present
                disjoint = self.node_time_dict is not None
                out = torch.ops.pyg.hetero_neighbor_sample_cpu(
                    self.node_types,
                    self.edge_types,
                    self.colptr_dict,
                    self.row_dict,
                    seed,  # seed_dict
                    self.num_neighbors,
                    kwargs.get('node_time_dict', self.node_time_dict),
                    True,  # csc
                    self.replace,
                    self.directed,
                    disjoint,
                    True,  # return_edge_id
                )
                row, col, node, edge, batch = out + (None, )
                if disjoint:
                    node = {k: v.t().contiguous() for k, v in node.items()}
                    batch = {k: v[0] for k, v in node.items()}
                    node = {k: v[1] for k, v in node.items()}

            else:
                if self.node_time_dict is None:
                    out = torch.ops.torch_sparse.hetero_neighbor_sample(
                        self.node_types,
                        self.edge_types,
                        self.colptr_dict,
                        self.row_dict,
                        seed,  # seed_dict
                        self.num_neighbors,
                        self.num_hops,
                        self.replace,
                        self.directed,
                    )
                else:
                    fn = torch.ops.torch_sparse.hetero_temporal_neighbor_sample
                    out = fn(
                        self.node_types,
                        self.edge_types,
                        self.colptr_dict,
                        self.row_dict,
                        seed,  # seed_dict
                        self.num_neighbors,
                        kwargs.get('node_time_dict', self.node_time_dict),
                        self.num_hops,
                        self.replace,
                        self.directed,
                    )
                node, row, col, edge, batch = out + (None, )

            return HeteroSamplerOutput(
                node=node,
                row=remap_keys(row, self.to_edge_type),
                col=remap_keys(col, self.to_edge_type),
                edge=remap_keys(edge, self.to_edge_type),
                batch=batch,
            )

        if issubclass(self.data_cls, Data):
            if _WITH_PYG_LIB:
                # TODO (matthias) Add `disjoint` option to `NeighborSampler`
                # TODO (matthias) `return_edge_id` if edge features present
                disjoint = self.node_time is not None
                out = torch.ops.pyg.neighbor_sample(
                    self.colptr,
                    self.row,
                    seed,  # seed
                    self.num_neighbors,
                    kwargs.get('node_time', self.node_time),
                    True,  # csc
                    self.replace,
                    self.directed,
                    disjoint,
                    True,  # return_edge_id
                )
                row, col, node, edge, batch = out + (None, )
                if disjoint:
                    batch, node = node.t().contiguous()

            else:
                if self.node_time is not None:
                    raise ValueError("'time_attr' not supported for "
                                     "neighbor sampling via 'torch-sparse'")
                out = torch.ops.torch_sparse.neighbor_sample(
                    self.colptr,
                    self.row,
                    seed,  # seed
                    self.num_neighbors,
                    self.replace,
                    self.directed,
                )
                node, row, col, edge, batch = out + (None, )

            return SamplerOutput(
                node=node,
                row=row,
                col=col,
                edge=edge,
                batch=batch,
            )

        raise TypeError(f"'{self.__class__.__name__}'' found invalid "
                        f"type: '{self.data_cls}'")

    # Node-based sampling #####################################################

    def sample_from_nodes(
        self,
        index: NodeSamplerInput,
        **kwargs,
    ) -> Union[SamplerOutput, HeteroSamplerOutput]:
        if isinstance(index, (list, tuple)):
            index = torch.tensor(index)

        # Tuple[FeatureStore, GraphStore] currently only supports heterogeneous
        # sampling:
        if self.data_cls == 'custom' or issubclass(self.data_cls, HeteroData):
            output = self._sample(seed={self.input_type: index})
            output.metadata = index.numel()

        elif issubclass(self.data_cls, Data):
            output = self._sample(seed=index)
            output.metadata = index.numel()

        else:
            raise TypeError(f"'{self.__class__.__name__}'' found invalid "
                            f"type: '{self.data_cls}'")

        return output

    # Edge-based sampling #####################################################

    def sample_from_edges(
        self,
        index: EdgeSamplerInput,
        **kwargs,
    ) -> Union[SamplerOutput, HeteroSamplerOutput]:
        negative_sampling_ratio = kwargs.get('negative_sampling_ratio', 0.0)
        query = [torch.stack(s, dim=0) for s in zip(*index)]
        edge_label_index = torch.stack(query[:2], dim=0)
        edge_label = query[2]
        edge_label_time = query[3] if len(query) == 4 else None

        out = add_negative_samples(edge_label_index, edge_label,
                                   edge_label_time, self.num_src_nodes,
                                   self.num_dst_nodes, negative_sampling_ratio)
        edge_label_index, edge_label, edge_label_time = out

        orig_edge_label_index = edge_label_index
        if (self.data_cls == 'custom'
                or issubclass(self.data_cls, HeteroData)):
            if self.input_type[0] != self.input_type[-1]:
                query_src = edge_label_index[0]
                query_src, reverse_src = query_src.unique(return_inverse=True)
                query_dst = edge_label_index[1]
                query_dst, reverse_dst = query_dst.unique(return_inverse=True)
                edge_label_index = torch.stack([reverse_src, reverse_dst], 0)
                query_node_dict = {
                    self.input_type[0]: query_src,
                    self.input_type[-1]: query_dst,
                }
            else:  # Merge both source and destination node indices:
                query_nodes = edge_label_index.view(-1)
                query_nodes, reverse = query_nodes.unique(return_inverse=True)
                edge_label_index = reverse.view(2, -1)
                query_node_dict = {self.input_type[0]: query_nodes}

            node_time_dict = self.node_time_dict
            if edge_label_time is not None:
                node_time_dict = set_node_time_dict(
                    node_time_dict, self.input_type, orig_edge_label_index,
                    edge_label_time, self.num_src_nodes, self.num_dst_nodes)

            output = self._sample(seed=query_node_dict,
                                  node_time_dict=node_time_dict)
            output.metadata = (edge_label_index, edge_label, edge_label_time)

        elif issubclass(self.data_cls, Data):
            query_nodes = edge_label_index.view(-1)
            query_nodes, reverse = query_nodes.unique(return_inverse=True)
            edge_label_index = reverse.view(2, -1)

            output = self._sample(seed=query_nodes)
            output.metadata = (edge_label_index, edge_label)

        else:
            raise TypeError(f"'{self.__class__.__name__}'' found invalid "
                            f"type: '{self.data_cls}'")

        return output

    # Other Utilities #########################################################

    @property
    def edge_permutation(self) -> Union[OptTensor, Dict[EdgeType, OptTensor]]:
        return self.perm
