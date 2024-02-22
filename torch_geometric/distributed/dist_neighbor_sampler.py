import itertools
import logging
import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.multiprocessing as mp
from torch import Tensor

from torch_geometric.distributed import (
    DistContext,
    LocalFeatureStore,
    LocalGraphStore,
)
from torch_geometric.distributed.event_loop import (
    ConcurrentEventLoop,
    to_asyncio_future,
)
from torch_geometric.distributed.rpc import (
    RPCCallBase,
    RPCRouter,
    rpc_async,
    rpc_partition_to_workers,
    rpc_register,
)
from torch_geometric.distributed.utils import (
    BatchDict,
    DistEdgeHeteroSamplerInput,
    NodeDict,
    remove_duplicates,
)
from torch_geometric.sampler import (
    EdgeSamplerInput,
    HeteroSamplerOutput,
    NegativeSampling,
    NeighborSampler,
    NodeSamplerInput,
    SamplerOutput,
)
from torch_geometric.sampler.base import NumNeighbors, SubgraphType
from torch_geometric.sampler.neighbor_sampler import neg_sample
from torch_geometric.sampler.utils import remap_keys
from torch_geometric.typing import EdgeType, NodeType

NumNeighborsType = Union[NumNeighbors, List[int], Dict[EdgeType, List[int]]]


class RPCSamplingCallee(RPCCallBase):
    r"""A wrapper for RPC callee that will perform RPC sampling from remote
    processes.
    """
    def __init__(self, sampler: NeighborSampler):
        super().__init__()
        self.sampler = sampler

    def rpc_async(self, *args, **kwargs) -> Any:
        return self.sampler._sample_one_hop(*args, **kwargs)

    def rpc_sync(self, *args, **kwargs) -> Any:
        pass


class DistNeighborSampler:
    r"""An implementation of a distributed and asynchronised neighbor sampler
    used by :class:`~torch_geometric.distributed.DistNeighborLoader` and
    :class:`~torch_geometric.distributed.DistLinkNeighborLoader`.
    """
    def __init__(
        self,
        current_ctx: DistContext,
        data: Tuple[LocalFeatureStore, LocalGraphStore],
        num_neighbors: NumNeighborsType,
        channel: Optional[mp.Queue] = None,
        replace: bool = False,
        subgraph_type: Union[SubgraphType, str] = 'directional',
        disjoint: bool = False,
        temporal_strategy: str = 'uniform',
        time_attr: Optional[str] = None,
        concurrency: int = 1,
        device: Optional[torch.device] = None,
        **kwargs,
    ):
        self.current_ctx = current_ctx

        self.feature_store, self.graph_store = data
        assert isinstance(self.graph_store, LocalGraphStore)
        assert isinstance(self.feature_store, LocalFeatureStore)
        self.is_hetero = self.graph_store.meta['is_hetero']

        self.num_neighbors = num_neighbors
        self.channel = channel
        self.concurrency = concurrency
        self.device = device
        self.event_loop = None
        self.replace = replace
        self.subgraph_type = SubgraphType(subgraph_type)
        self.disjoint = disjoint
        self.temporal_strategy = temporal_strategy
        self.time_attr = time_attr
        self.temporal = time_attr is not None
        self.with_edge_attr = self.feature_store.has_edge_attr()
        self.csc = True

    def init_sampler_instance(self):
        self._sampler = NeighborSampler(
            data=(self.feature_store, self.graph_store),
            num_neighbors=self.num_neighbors,
            subgraph_type=self.subgraph_type,
            replace=self.replace,
            disjoint=self.disjoint,
            temporal_strategy=self.temporal_strategy,
            time_attr=self.time_attr,
        )

        self.num_hops = self._sampler.num_neighbors.num_hops
        self.node_types = self._sampler.node_types
        self.edge_types = self._sampler.edge_types
        self.node_time = self._sampler.node_time
        self.edge_time = self._sampler.edge_time

    def register_sampler_rpc(self) -> None:
        partition2workers = rpc_partition_to_workers(
            current_ctx=self.current_ctx,
            num_partitions=self.graph_store.num_partitions,
            current_partition_idx=self.graph_store.partition_idx,
        )
        self.rpc_router = RPCRouter(partition2workers)
        self.feature_store.set_rpc_router(self.rpc_router)

        rpc_sample_callee = RPCSamplingCallee(self)
        self.rpc_sample_callee_id = rpc_register(rpc_sample_callee)

    def init_event_loop(self) -> None:
        if self.event_loop is None:
            self.event_loop = ConcurrentEventLoop(self.concurrency)
            self.event_loop.start_loop()
            logging.info(f'{self} uses {self.event_loop}')

    # Node-based distributed sampling #########################################

    def sample_from_nodes(
        self,
        inputs: NodeSamplerInput,
        **kwargs,
    ) -> Optional[Union[SamplerOutput, HeteroSamplerOutput]]:
        self.init_event_loop()

        inputs = NodeSamplerInput.cast(inputs)
        if self.channel is None:
            # synchronous sampling
            return self.event_loop.run_task(
                coro=self._sample_from(self.node_sample, inputs))

        # asynchronous sampling
        cb = kwargs.get("callback", None)
        self.event_loop.add_task(
            coro=self._sample_from(self.node_sample, inputs), callback=cb)
        return None

    # Edge-based distributed sampling #########################################

    def sample_from_edges(
        self,
        inputs: EdgeSamplerInput,
        neg_sampling: Optional[NegativeSampling] = None,
        **kwargs,
    ) -> Optional[Union[SamplerOutput, HeteroSamplerOutput]]:
        self.init_event_loop()

        if self.channel is None:
            # synchronous sampling
            return self.event_loop.run_task(coro=self._sample_from(
                self.edge_sample, inputs, self.node_sample, self._sampler.
                num_nodes, self.disjoint, self.node_time, neg_sampling))

        # asynchronous sampling
        cb = kwargs.get("callback", None)
        self.event_loop.add_task(
            coro=self._sample_from(self.edge_sample, inputs, self.node_sample,
                                   self._sampler.num_nodes, self.disjoint,
                                   self.node_time, neg_sampling), callback=cb)
        return None

    async def _sample_from(
        self,
        async_func,
        *args,
        **kwargs,
    ) -> Optional[Union[SamplerOutput, HeteroSamplerOutput]]:
        sampler_output = await async_func(*args, **kwargs)

        if self.subgraph_type == SubgraphType.bidirectional:
            sampler_output = sampler_output.to_bidirectional()

        res = await self._collate_fn(sampler_output)

        if self.channel is None:
            return res
        self.channel.put(res)
        return None

    async def node_sample(
        self,
        inputs: Union[NodeSamplerInput, DistEdgeHeteroSamplerInput],
    ) -> Union[SamplerOutput, HeteroSamplerOutput]:
        r"""Performs layer-by-layer distributed sampling from a
        :class:`NodeSamplerInput` or :class:`DistEdgeHeteroSamplerInput` and
        returns the output of the sampling procedure.

        .. note::
            In case of distributed training it is required to synchronize the
            results between machines after each layer.
        """
        input_type = inputs.input_type
        self.input_type = input_type

        if isinstance(inputs, NodeSamplerInput):
            seed = inputs.node.to(self.device)
            batch_size = len(inputs.node)
            seed_batch = torch.arange(batch_size) if self.disjoint else None

            metadata = (inputs.input_id, inputs.time, batch_size)

            seed_time: Optional[Tensor] = None
            if self.temporal:
                if inputs.time is not None:
                    seed_time = inputs.time.to(self.device)
                elif self.node_time is not None:
                    if not self.is_hetero:
                        seed_time = self.node_time[seed]
                    else:
                        seed_time = self.node_time[input_type][seed]
                else:
                    raise ValueError("Seed time needs to be specified")
        else:  # `DistEdgeHeteroSamplerInput`:
            metadata = None  # Metadata is added during `edge_sample`.

        # Heterogeneous Neighborhood Sampling #################################

        if self.is_hetero:
            if input_type is None:
                raise ValueError("Input type should be defined")

            node_dict = NodeDict(self.node_types, self.num_hops)
            batch_dict = BatchDict(self.node_types, self.num_hops)

            if isinstance(inputs, NodeSamplerInput):
                seed_dict: Dict[NodeType, Tensor] = {input_type: seed}
                if self.temporal:
                    node_dict.seed_time[input_type][0] = seed_time.clone()

            else:  # `DistEdgeHeteroSamplerInput`:
                seed_dict = inputs.node_dict
                if self.temporal:
                    for k, v in inputs.node_dict.items():
                        if inputs.time_dict is not None:
                            node_dict.seed_time[k][0] = inputs.time_dict[k]
                        elif self.node_time is not None:
                            node_dict.seed_time[k][0] = self.node_time[k][v]
                        else:
                            raise ValueError("Seed time needs to be specified")

            edge_dict: Dict[EdgeType, Tensor] = {
                k: torch.empty(0, dtype=torch.int64)
                for k in self.edge_types
            }
            sampled_nbrs_per_node_dict: Dict[EdgeType, List[List]] = {
                k: [[] for _ in range(self.num_hops)]
                for k in self.edge_types
            }
            num_sampled_edges_dict: Dict[EdgeType, List[int]] = {
                k: []
                for k in self.edge_types
            }
            num_sampled_nodes_dict: Dict[NodeType, List[int]] = {
                k: [0]
                for k in self.node_types
            }

            # Fill in node_dict and batch_dict with input data:
            batch_len = 0
            for k, v in seed_dict.items():
                node_dict.src[k][0] = v
                node_dict.out[k] = v
                num_sampled_nodes_dict[k][0] = len(v)

                if self.disjoint:
                    src_batch = torch.arange(batch_len, batch_len + len(v))
                    batch_dict.src[k][0] = src_batch
                    batch_dict.out[k] = src_batch

                    batch_len = len(src_batch)

            # Loop over the layers:
            for i in range(self.num_hops):
                # Sample neighbors per edge type:
                for edge_type in self.edge_types:
                    # `src` is a destination node type of a given edge.
                    src = edge_type[0] if not self.csc else edge_type[2]

                    if node_dict.src[src][i].numel() == 0:
                        # No source nodes of this type in the current layer.
                        num_sampled_edges_dict[edge_type].append(0)
                        continue

                    if isinstance(self.num_neighbors, list):
                        one_hop_num = self.num_neighbors[i]
                    else:
                        one_hop_num = self.num_neighbors[edge_type][i]

                    # Sample neighbors:
                    out = await self.sample_one_hop(
                        node_dict.src[src][i],
                        one_hop_num,
                        node_dict.seed_time[src][i],
                        batch_dict.src[src][i],
                        edge_type,
                    )

                    if out.node.numel() == 0:  # No neighbors were sampled.
                        num_sampled_edges_dict[edge_type].append(0)
                        continue

                    # `dst` is a destination node type of a given edge.
                    dst = edge_type[2] if not self.csc else edge_type[0]

                    # Remove duplicates:
                    (
                        src_node,
                        node_dict.out[dst],
                        src_batch,
                        batch_dict.out[dst],
                    ) = remove_duplicates(
                        out,
                        node_dict.out[dst],
                        batch_dict.out[dst],
                        self.disjoint,
                    )

                    # Create src nodes for the next layer:
                    node_dict.src[dst][i + 1] = torch.cat(
                        [node_dict.src[dst][i + 1], src_node])
                    if self.disjoint:
                        batch_dict.src[dst][i + 1] = torch.cat(
                            [batch_dict.src[dst][i + 1], src_batch])

                    # Save sampled nodes with duplicates to be able to create
                    # local edge indices:
                    node_dict.with_dupl[dst] = torch.cat(
                        [node_dict.with_dupl[dst], out.node])

                    edge_dict[edge_type] = torch.cat(
                        [edge_dict[edge_type], out.edge])

                    if self.disjoint:
                        batch_dict.with_dupl[dst] = torch.cat(
                            [batch_dict.with_dupl[dst], out.batch])

                    if self.temporal and i < self.num_hops - 1:
                        # Assign seed time based on source node subgraph ID:
                        if isinstance(inputs, NodeSamplerInput):
                            src_seed_time = [
                                seed_time[(seed_batch == batch_idx).nonzero()]
                                for batch_idx in src_batch
                            ]
                            src_seed_time = torch.as_tensor(
                                src_seed_time, dtype=torch.int64)

                        else:  # `DistEdgeHeteroSamplerInput`:
                            src_seed_time = torch.empty(0, dtype=torch.int64)
                            for k, v in batch_dict.src.items():
                                time = [
                                    node_dict.seed_time[k][0][(
                                        v[0] == batch_idx).nonzero()]
                                    for batch_idx in src_batch
                                ]
                                try:
                                    time = torch.as_tensor(
                                        time, dtype=torch.int64)
                                    src_seed_time = torch.cat(
                                        [src_seed_time, time])
                                except Exception:
                                    # `time`  may be an empty tensors, because
                                    # no nodes of this type were sampled.
                                    pass

                        node_dict.seed_time[dst][i + 1] = torch.cat(
                            [node_dict.seed_time[dst][i + 1], src_seed_time])

                    # Collect sampled neighbors per node for each layer:
                    sampled_nbrs_per_node_dict[edge_type][i] += out.metadata[0]

                    num_sampled_edges_dict[edge_type].append(len(out.node))

                for node_type in self.node_types:
                    num_sampled_nodes_dict[node_type].append(
                        len(node_dict.src[node_type][i + 1]))

            sampled_nbrs_per_node_dict = remap_keys(sampled_nbrs_per_node_dict,
                                                    self._sampler.to_rel_type)

            # Create local edge indices for a batch:
            row_dict, col_dict = torch.ops.pyg.hetero_relabel_neighborhood(
                self.node_types,
                self.edge_types,
                seed_dict,
                node_dict.with_dupl,
                sampled_nbrs_per_node_dict,
                self._sampler.num_nodes,
                batch_dict.with_dupl,
                self.csc,
                self.disjoint,
            )

            sampler_output = HeteroSamplerOutput(
                node=node_dict.out,
                row=remap_keys(row_dict, self._sampler.to_edge_type),
                col=remap_keys(col_dict, self._sampler.to_edge_type),
                edge=edge_dict,
                batch=batch_dict.out if self.disjoint else None,
                num_sampled_nodes=num_sampled_nodes_dict,
                num_sampled_edges=num_sampled_edges_dict,
                metadata=metadata,
            )

        # Homogeneous Neighborhood Sampling ###################################

        else:
            src = seed
            node = src.clone()

            src_batch = seed_batch.clone() if self.disjoint else None
            batch = seed_batch.clone() if self.disjoint else None

            src_seed_time = seed_time.clone() if self.temporal else None

            node_with_dupl = [torch.empty(0, dtype=torch.int64)]
            batch_with_dupl = [torch.empty(0, dtype=torch.int64)]
            edge = [torch.empty(0, dtype=torch.int64)]

            sampled_nbrs_per_node = []
            num_sampled_nodes = [seed.numel()]
            num_sampled_edges = []

            # Loop over the layers:
            for i, one_hop_num in enumerate(self.num_neighbors):
                out = await self.sample_one_hop(src, one_hop_num,
                                                src_seed_time, src_batch)
                if out.node.numel() == 0:
                    # No neighbors were sampled:
                    num_zero_layers = self.num_hops - i
                    num_sampled_nodes += num_zero_layers * [0]
                    num_sampled_edges += num_zero_layers * [0]
                    break

                # Remove duplicates:
                src, node, src_batch, batch = remove_duplicates(
                    out, node, batch, self.disjoint)

                node_with_dupl.append(out.node)
                edge.append(out.edge)

                if self.disjoint:
                    batch_with_dupl.append(out.batch)

                if self.temporal and i < self.num_hops - 1:
                    # Assign seed time based on src nodes subgraph IDs.
                    src_seed_time = [
                        seed_time[(seed_batch == batch_idx).nonzero()]
                        for batch_idx in src_batch
                    ]
                    src_seed_time = torch.as_tensor(src_seed_time,
                                                    dtype=torch.int64)

                num_sampled_nodes.append(len(src))
                num_sampled_edges.append(len(out.node))
                sampled_nbrs_per_node += out.metadata[0]

            row, col = torch.ops.pyg.relabel_neighborhood(
                seed,
                torch.cat(node_with_dupl),
                sampled_nbrs_per_node,
                self._sampler.num_nodes,
                torch.cat(batch_with_dupl) if self.disjoint else None,
                self.csc,
                self.disjoint,
            )

            sampler_output = SamplerOutput(
                node=node,
                row=row,
                col=col,
                edge=torch.cat(edge),
                batch=batch if self.disjoint else None,
                num_sampled_nodes=num_sampled_nodes,
                num_sampled_edges=num_sampled_edges,
                metadata=metadata,
            )

        return sampler_output

    async def edge_sample(
        self,
        inputs: EdgeSamplerInput,
        sample_fn: Callable,
        num_nodes: Union[int, Dict[NodeType, int]],
        disjoint: bool,
        node_time: Optional[Union[Tensor, Dict[str, Tensor]]] = None,
        neg_sampling: Optional[NegativeSampling] = None,
    ) -> Union[SamplerOutput, HeteroSamplerOutput]:
        r"""Performs layer-by-layer distributed sampling from an
        :class:`EdgeSamplerInput` and returns the output of the sampling
        procedure.

        .. note::
            In case of distributed training it is required to synchronize the
            results between machines after each layer.
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

        # Negative Sampling ###################################################

        if neg_sampling is not None:
            # When we are doing negative sampling, we append negative
            # information of nodes/edges to `src`, `dst`, `src_time`,
            # `dst_time`. Later on, we can easily reconstruct what belongs to
            # positive and negative examples by slicing via `num_pos`.
            num_neg = math.ceil(num_pos * neg_sampling.amount)

            if neg_sampling.is_binary():
                # In the "binary" case, we randomly sample negative pairs of
                # nodes.
                if isinstance(node_time, dict):
                    src_node_time = node_time.get(input_type[0])
                else:
                    src_node_time = node_time

                src_neg = neg_sample(src, neg_sampling, num_src_nodes,
                                     src_time, src_node_time)
                src = torch.cat([src, src_neg], dim=0)

                if isinstance(node_time, dict):
                    dst_node_time = node_time.get(input_type[-1])
                else:
                    dst_node_time = node_time

                dst_neg = neg_sample(dst, neg_sampling, num_dst_nodes,
                                     dst_time, dst_node_time)
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
                # In the "triplet" case, we randomly sample negative
                # destinations.
                if isinstance(node_time, dict):
                    dst_node_time = node_time.get(input_type[-1])
                else:
                    dst_node_time = node_time

                dst_neg = neg_sample(dst, neg_sampling, num_dst_nodes,
                                     dst_time, dst_node_time)
                dst = torch.cat([dst, dst_neg], dim=0)

                assert edge_label is None

                if edge_label_time is not None:
                    dst_time = edge_label_time.repeat(1 + neg_sampling.amount)

        # Heterogeneus Neighborhood Sampling ##################################

        if input_type is not None:
            if input_type[0] != input_type[-1]:  # Two distinct node types:

                if not disjoint:
                    src, inverse_src = src.unique(return_inverse=True)
                    dst, inverse_dst = dst.unique(return_inverse=True)

                seed_dict = {input_type[0]: src, input_type[-1]: dst}

                seed_time_dict = None
                if edge_label_time is not None:  # Always disjoint.
                    seed_time_dict = {
                        input_type[0]: src_time,
                        input_type[-1]: dst_time,
                    }

                out = await sample_fn(
                    DistEdgeHeteroSamplerInput(
                        input_id=inputs.input_id,
                        node_dict=seed_dict,
                        time_dict=seed_time_dict,
                        input_type=input_type,
                    ))

            else:
                # Only a single node type: Merge both source and destination.
                seed = torch.cat([src, dst], dim=0)

                if not disjoint:
                    seed, inverse_seed = seed.unique(return_inverse=True)

                seed_dict = {input_type[0]: seed}

                seed_time = None
                if edge_label_time is not None:  # Always disjoint.
                    seed_time = torch.cat([src_time, dst_time], dim=0)

                out = await sample_fn(
                    NodeSamplerInput(
                        input_id=inputs.input_id,
                        node=seed,
                        time=seed_time,
                        input_type=input_type[0],
                    ))

            # Enhance `out` by label information ##############################
            if disjoint:
                for key, batch in out.batch.items():
                    out.batch[key] = batch % num_pos

            if neg_sampling is None or neg_sampling.is_binary():
                if disjoint:
                    if input_type[0] != input_type[-1]:
                        edge_label_index = torch.arange(num_pos + num_neg)
                        edge_label_index = edge_label_index.repeat(2)
                        edge_label_index = edge_label_index.view(2, -1)
                    else:
                        num_labels = num_pos + num_neg
                        edge_label_index = torch.arange(2 * (num_labels))
                        edge_label_index = edge_label_index.view(2, -1)
                else:
                    if input_type[0] != input_type[-1]:
                        edge_label_index = torch.stack([
                            inverse_src,
                            inverse_dst,
                        ], dim=0)
                    else:
                        edge_label_index = inverse_seed.view(2, -1)

                out.metadata = (input_id, edge_label_index, edge_label,
                                src_time)

            elif neg_sampling.is_triplet():
                if disjoint:
                    src_index = torch.arange(num_pos)
                    if input_type[0] != input_type[-1]:
                        dst_pos_index = torch.arange(num_pos)
                        # `dst_neg_index` needs to be offset such that indices
                        # with offset `num_pos` belong to the same triplet:
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

        # Homogeneous Neighborhood Sampling ###################################

        else:

            seed = torch.cat([src, dst], dim=0)
            seed_time = None

            if not disjoint:
                seed, inverse_seed = seed.unique(return_inverse=True)

            if edge_label_time is not None:  # Always disjoint.
                seed_time = torch.cat([src_time, dst_time])

            out = await sample_fn(
                NodeSamplerInput(
                    input_id=inputs.input_id,
                    node=seed,
                    time=seed_time,
                    input_type=None,
                ))

            # Enhance `out` by label information ##############################
            if neg_sampling is None or neg_sampling.is_binary():
                if disjoint:
                    out.batch = out.batch % num_pos
                    edge_label_index = torch.arange(seed.numel()).view(2, -1)
                else:
                    edge_label_index = inverse_seed.view(2, -1)

                out.metadata = (input_id, edge_label_index, edge_label,
                                src_time)

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

    def _get_sampler_output(
        self,
        outputs: List[SamplerOutput],
        seed_size: int,
        p_id: int,
        src_batch: Optional[Tensor] = None,
    ) -> SamplerOutput:
        r"""Used when seed nodes belongs to one partition. It's purpose is to
        remove seed nodes from sampled nodes and calculates how many neighbors
        were sampled by each src node based on the
        :obj:`cumsum_neighbors_per_node`. Returns updated sampler output.
        """
        cumsum_neighbors_per_node = outputs[p_id].metadata[0]

        # do not include seed
        outputs[p_id].node = outputs[p_id].node[seed_size:]

        begin = np.array(cumsum_neighbors_per_node[1:])
        end = np.array(cumsum_neighbors_per_node[:-1])

        sampled_nbrs_per_node = list(np.subtract(begin, end))

        outputs[p_id].metadata = (sampled_nbrs_per_node, )

        if self.disjoint:
            batch = [[src_batch[i]] * nbrs_per_node
                     for i, nbrs_per_node in enumerate(sampled_nbrs_per_node)]
            outputs[p_id].batch = Tensor(
                list(itertools.chain.from_iterable(batch))).type(torch.int64)

        return outputs[p_id]

    def _merge_sampler_outputs(
        self,
        partition_ids: Tensor,
        partition_orders: Tensor,
        outputs: List[SamplerOutput],
        one_hop_num: int,
        src_batch: Optional[Tensor] = None,
    ) -> SamplerOutput:
        r"""Merges samplers outputs from different partitions, so that they
        are sorted according to the sampling order. Removes seed nodes from
        sampled nodes and calculates how many neighbors were sampled by each
        src node based on the :obj:`cumsum_neighbors_per_node`. Leverages the
        :obj:`pyg-lib` :meth:`merge_sampler_outputs` function.

        Args:
            partition_ids (torch.Tensor): Contains information on which
                partition seeds nodes are located on.
            partition_orders (torch.Tensor): Contains information about the
                order of seed nodes in each partition.
            outputs (List[SamplerOutput]): List of all samplers outputs.
            one_hop_num (int): Max number of neighbors sampled in the current
                layer.
            src_batch (torch.Tensor, optional): The batch assignment of seed
                nodes. (default: :obj:`None`)

        Returns :obj:`SamplerOutput` containing all merged outputs.
        """
        sampled_nodes_with_dupl = [
            o.node if o is not None else torch.empty(0, dtype=torch.int64)
            for o in outputs
        ]
        edge_ids = [
            o.edge if o is not None else torch.empty(0, dtype=torch.int64)
            for o in outputs
        ]
        cumm_sampled_nbrs_per_node = [
            o.metadata[0] if o is not None else [] for o in outputs
        ]

        partition_ids = partition_ids.tolist()
        partition_orders = partition_orders.tolist()

        partitions_num = self.graph_store.meta["num_parts"]

        out = torch.ops.pyg.merge_sampler_outputs(
            sampled_nodes_with_dupl,
            edge_ids,
            cumm_sampled_nbrs_per_node,
            partition_ids,
            partition_orders,
            partitions_num,
            one_hop_num,
            src_batch,
            self.disjoint,
        )
        (
            out_node_with_dupl,
            out_edge,
            out_batch,
            out_sampled_nbrs_per_node,
        ) = out

        return SamplerOutput(
            out_node_with_dupl,
            None,
            None,
            out_edge,
            out_batch if self.disjoint else None,
            metadata=(out_sampled_nbrs_per_node, ),
        )

    async def sample_one_hop(
        self,
        srcs: Tensor,
        one_hop_num: int,
        seed_time: Optional[Tensor] = None,
        src_batch: Optional[Tensor] = None,
        edge_type: Optional[EdgeType] = None,
    ) -> SamplerOutput:
        r"""Samples one-hop neighbors for a set of seed nodes in :obj:`srcs`.
        If seed nodes are located on a local partition, evaluates the sampling
        function on the current machine. If seed nodes are from a remote
        partition, sends a request to a remote machine that contains this
        partition.
        """
        src_node_type = None if not self.is_hetero else edge_type[2]
        partition_ids = self.graph_store.get_partition_ids_from_nids(
            srcs, src_node_type)
        partition_orders = torch.zeros(len(partition_ids), dtype=torch.long)

        p_outputs: List[SamplerOutput] = [
            None
        ] * self.graph_store.meta["num_parts"]
        futs: List[torch.futures.Future] = []

        local_only = True
        single_partition = len(set(partition_ids.tolist())) == 1

        for i in range(self.graph_store.num_partitions):
            p_id = (self.graph_store.partition_idx +
                    i) % self.graph_store.num_partitions
            p_mask = partition_ids == p_id
            p_srcs = torch.masked_select(srcs, p_mask)
            p_seed_time = (torch.masked_select(seed_time, p_mask)
                           if self.temporal else None)

            p_indices = torch.arange(len(p_srcs), dtype=torch.long)
            partition_orders[p_mask] = p_indices

            if p_srcs.shape[0] > 0:
                if p_id == self.graph_store.partition_idx:
                    # Sample for one hop on a local machine:
                    p_nbr_out = self._sample_one_hop(p_srcs, one_hop_num,
                                                     p_seed_time, edge_type)
                    p_outputs.pop(p_id)
                    p_outputs.insert(p_id, p_nbr_out)

                else:  # Sample on a remote machine:
                    local_only = False
                    to_worker = self.rpc_router.get_to_worker(p_id)
                    futs.append(
                        rpc_async(
                            to_worker,
                            self.rpc_sample_callee_id,
                            args=(p_srcs, one_hop_num, p_seed_time, edge_type),
                        ))

        if not local_only:
            # Src nodes are remote
            res_fut_list = await to_asyncio_future(
                torch.futures.collect_all(futs))
            for i, res_fut in enumerate(res_fut_list):
                p_id = (self.graph_store.partition_idx + i +
                        1) % self.graph_store.num_partitions
                p_outputs.pop(p_id)
                p_outputs.insert(p_id, res_fut.wait())

        # All src nodes are in the same partition
        if single_partition:
            return self._get_sampler_output(p_outputs, len(srcs),
                                            partition_ids[0], src_batch)

        return self._merge_sampler_outputs(partition_ids, partition_orders,
                                           p_outputs, one_hop_num, src_batch)

    def _sample_one_hop(
        self,
        input_nodes: Tensor,
        num_neighbors: int,
        seed_time: Optional[Tensor] = None,
        edge_type: Optional[EdgeType] = None,
    ) -> SamplerOutput:
        r"""Implements one-hop neighbor sampling for a set of input nodes for a
        specific edge type.
        """
        if not self.is_hetero:
            colptr = self._sampler.colptr
            row = self._sampler.row
            node_time = self.node_time
            edge_time = self.edge_time
        else:
            # Given edge type, get input data and evaluate sample function:
            rel_type = '__'.join(edge_type)
            colptr = self._sampler.colptr_dict[rel_type]
            row = self._sampler.row_dict[rel_type]
            # `node_time` is a destination node time:
            node_time = (self.node_time or {}).get(edge_type[0], None)
            edge_time = (self.edge_time or {}).get(edge_type, None)

        out = torch.ops.pyg.dist_neighbor_sample(
            colptr,
            row,
            input_nodes.to(colptr.dtype),
            num_neighbors,
            node_time,
            edge_time,
            seed_time,
            None,  # TODO: edge_weight
            True,  # csc
            self.replace,
            self.subgraph_type != SubgraphType.induced,
            self.disjoint and self.temporal,
            self.temporal_strategy,
        )
        node, edge, cumsum_neighbors_per_node = out

        if self.disjoint and self.temporal:
            # We create a batch during the step of merging sampler outputs.
            _, node = node.t().contiguous()

        return SamplerOutput(
            node=node,
            row=None,
            col=None,
            edge=edge,
            batch=None,
            metadata=(cumsum_neighbors_per_node, ),
        )

    async def _collate_fn(
        self, output: Union[SamplerOutput, HeteroSamplerOutput]
    ) -> Union[SamplerOutput, HeteroSamplerOutput]:
        r"""Collect labels and features for the sampled subgrarph if necessary,
        and put them into a sample message.
        """
        if self.is_hetero:
            labels = {}
            nfeats = {}
            efeats = {}
            labels = self.feature_store.labels
            if labels is not None:
                if isinstance(self.input_type, tuple):  # Edge labels.
                    labels = {
                        self.input_type: labels[output.edge[self.input_type]]
                    }
                else:  # Node labels.
                    labels = {
                        self.input_type: labels[output.node[self.input_type]]
                    }
            # Collect node features.
            if output.node is not None:
                for ntype in output.node.keys():
                    if output.node[ntype].numel() > 0:
                        fut = self.feature_store.lookup_features(
                            is_node_feat=True,
                            index=output.node[ntype],
                            input_type=ntype,
                        )
                        nfeat = await to_asyncio_future(fut)
                        nfeat = nfeat.to(torch.device("cpu"))
                        nfeats[ntype] = nfeat
                    else:
                        nfeats[ntype] = None
            # Collect edge features
            if output.edge is not None and self.with_edge_attr:
                for edge_type in output.edge.keys():
                    if output.edge[edge_type].numel() > 0:
                        fut = self.feature_store.lookup_features(
                            is_node_feat=False,
                            index=output.edge[edge_type],
                            input_type=edge_type,
                        )
                        efeat = await to_asyncio_future(fut)
                        efeat = efeat.to(torch.device("cpu"))
                        efeats[edge_type] = efeat
                    else:
                        efeats[edge_type] = None

        else:  # Homogeneous:
            # Collect node labels.
            if self.feature_store.labels is not None:
                labels = self.feature_store.labels[output.node]
            else:
                labels = None
            # Collect node features.
            if output.node is not None:
                fut = self.feature_store.lookup_features(
                    is_node_feat=True, index=output.node)
                nfeats = await to_asyncio_future(fut)
                nfeats = nfeats.to(torch.device("cpu"))
            else:
                nfeats = None
            # Collect edge features.
            if output.edge is not None and self.with_edge_attr:
                fut = self.feature_store.lookup_features(
                    is_node_feat=False, index=output.edge)
                efeats = await to_asyncio_future(fut)
                efeats = efeats.to(torch.device("cpu"))
            else:
                efeats = None

        output.metadata = (*output.metadata, nfeats, labels, efeats)
        return output

    @property
    def edge_permutation(self) -> None:
        return None

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(pid={mp.current_process().pid})'
