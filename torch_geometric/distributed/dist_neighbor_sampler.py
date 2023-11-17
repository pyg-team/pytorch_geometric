import itertools
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.multiprocessing as mp
from torch import Tensor

from torch_geometric.distributed import LocalFeatureStore, LocalGraphStore
from torch_geometric.distributed.dist_context import DistContext, DistRole
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
    shutdown_rpc,
)
from torch_geometric.distributed.utils import (
    BatchDict,
    NodeDict,
    remove_duplicates,
)
from torch_geometric.sampler import (
    EdgeSamplerInput,
    HeteroSamplerOutput,
    NeighborSampler,
    NodeSamplerInput,
    SamplerOutput,
)
from torch_geometric.sampler.base import NumNeighbors, SubgraphType
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
    used by :class:`~torch_geometric.distributed.DistNeighborLoader`.
    """
    def __init__(
        self,
        current_ctx: DistContext,
        rpc_worker_names: Dict[DistRole, List[str]],
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
        self.rpc_worker_names = rpc_worker_names

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
        self.with_edge_attr = self.feature_store.has_edge_attr()
        self.csc = True

    def register_sampler_rpc(self) -> None:
        partition2workers = rpc_partition_to_workers(
            current_ctx=self.current_ctx,
            num_partitions=self.graph_store.num_partitions,
            current_partition_idx=self.graph_store.partition_idx,
        )
        self.rpc_router = RPCRouter(partition2workers)
        self.feature_store.set_rpc_router(self.rpc_router)

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

        rpc_sample_callee = RPCSamplingCallee(self)
        self.rpc_sample_callee_id = rpc_register(rpc_sample_callee)

    def init_event_loop(self) -> None:
        self.event_loop = ConcurrentEventLoop(self.concurrency)
        self.event_loop.start_loop()

    # Node-based distributed sampling #########################################

    def sample_from_nodes(
        self,
        inputs: NodeSamplerInput,
        **kwargs,
    ) -> Optional[Union[SamplerOutput, HeteroSamplerOutput]]:
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

    async def _sample_from(
        self,
        async_func,
        *args,
        **kwargs,
    ) -> Optional[Union[SamplerOutput, HeteroSamplerOutput]]:

        sampler_output = await async_func(*args, **kwargs)
        res = await self._collate_fn(sampler_output)

        if self.channel is None:
            return res
        self.channel.put(res)
        return None

    async def node_sample(
        self,
        inputs: Union[NodeSamplerInput, EdgeSamplerInput],
        dst_time: Tensor = None,
    ) -> Union[SamplerOutput, HeteroSamplerOutput]:
        r"""Performs layer by layer distributed sampling from a
        :class:`NodeSamplerInput` and returns the output of the sampling
        procedure.

        Note:
            In case of distributed training it is required to synchronize the
            results between machines after each layer.
        """
        input_type = inputs.input_type
        self.input_type = input_type
        batch_size = inputs.node.size()[0]

        seed_dict = None
        seed_time_dict = None
        src_batch_dict = None

        if isinstance(inputs, NodeSamplerInput):
            seed = inputs.node.to(self.device)
            seed_time = None
            if self.time_attr is not None:
                if inputs.time is not None:
                    seed_time = inputs.time.to(self.device)
                else:
                    seed_time = self.node_time[seed]
            src_batch = torch.arange(batch_size) if self.disjoint else None
            metadata = (seed, seed_time)

        elif isinstance(inputs, EdgeSamplerInput) and self.is_hetero:
            seed_dict = {input_type[0]: inputs.row, input_type[-1]: inputs.col}
            if dst_time is not None:
                seed_time_dict = {
                    input_type[0]: inputs.time,
                    input_type[-1]: dst_time,
                }

            if self.disjoint:
                src_batch_dict = {
                    input_type[0]: torch.arange(batch_size),
                    input_type[-1]: torch.arange(batch_size, batch_size * 2),
                }
            metadata = (seed_dict, seed_time_dict)

        else:
            raise NotImplementedError

        if self.is_hetero:
            if input_type is None:
                raise ValueError("Input type should be defined")

            seed_dict: Dict[NodeType, Tensor] = {input_type: seed}
            seed_time_dict: Dict[NodeType, Tensor] = {input_type: seed_time}

            node_dict = NodeDict()
            batch_dict = BatchDict(self.disjoint)
            edge_dict: Dict[EdgeType, Tensor] = {}
            num_sampled_nodes_dict: Dict[NodeType, List[int]] = {}
            sampled_nbrs_per_node_dict: Dict[EdgeType, List[int]] = {}
            num_sampled_edges_dict: Dict[EdgeType, List[int]] = {}

            for ntype in self._sampler.node_types:
                num_sampled_nodes_dict.update({ntype: [0]})

            for etype in self._sampler.edge_types:
                edge_dict.update({etype: torch.empty(0, dtype=torch.int64)})
                num_sampled_edges_dict.update({etype: []})
                sampled_nbrs_per_node_dict.update({etype: []})

            if isinstance(inputs, EdgeSamplerInput):
                node_dict.src = seed_dict
                node_dict.out = {
                    input_type[0]: inputs.row.numpy(),
                    input_type[-1]: inputs.col.numpy(),
                }

                num_sampled_nodes_dict = {
                    input_type[0]: [inputs.row.numel()],
                    input_type[-1]: [inputs.col.numel()],
                }

                if self.disjoint:
                    batch_dict = src_batch_dict
                    batch_dict.out = {
                        input_type[0]: src_batch_dict[input_type[0]].numpy(),
                        input_type[-1]: src_batch_dict[input_type[-1]].numpy(),
                    }

            else:
                node_dict.src[input_type] = seed
                node_dict.out[input_type] = seed.numpy()

                num_sampled_nodes_dict[input_type].append(seed.numel())

                if self.disjoint:
                    batch_dict.src[input_type] = src_batch
                    batch_dict.out[input_type] = src_batch.numpy()

            # loop over the layers
            for i in range(self._sampler.num_hops):
                # create tasks
                task_dict = {}
                for etype in self._sampler.edge_types:
                    src = etype[0] if not self.csc else etype[2]

                    if node_dict.src[src].numel():
                        seed_time = (seed_time_dict.get(src, None)
                                     if seed_time_dict is not None else None)
                        if isinstance(self.num_neighbors, list):
                            one_hop_num = self.num_neighbors[i]
                        else:
                            one_hop_num = self.num_neighbors[etype][i]

                        task_dict[etype] = self.event_loop._loop.create_task(
                            self.sample_one_hop(
                                node_dict.src[src],
                                one_hop_num,
                                seed_time,
                                batch_dict.src[src],
                                etype,
                            ))

                for etype, task in task_dict.items():
                    out: HeteroSamplerOutput = await task

                    if out.node.numel() == 0:
                        # no neighbors were sampled
                        break

                    dst = etype[2] if not self.csc else etype[0]

                    # remove duplicates
                    (
                        node_dict.src[dst],
                        node_dict.out[dst],
                        batch_dict.src[dst],
                        batch_dict.out[dst],
                    ) = remove_duplicates(
                        out,
                        node_dict.out[dst],
                        batch_dict.out[dst],
                        self.disjoint,
                    )

                    node_dict.with_dupl[dst] = torch.cat(
                        [node_dict.with_dupl[dst], out.node])
                    edge_dict[etype] = torch.cat([edge_dict[etype], out.edge])

                    if self.disjoint:
                        batch_dict.with_dupl[dst] = torch.cat(
                            [batch_dict.with_dupl[dst], out.batch])

                    num_sampled_nodes_dict[dst].append(len(node_dict.src[dst]))
                    num_sampled_edges_dict[etype].append(len(out.node))
                    sampled_nbrs_per_node_dict[etype] += out.metadata

            sampled_nbrs_per_node_dict = remap_keys(sampled_nbrs_per_node_dict,
                                                    self._sampler.to_rel_type)

            row_dict, col_dict = torch.ops.pyg.hetero_relabel_neighborhood(
                self._sampler.node_types,
                self._sampler.edge_types,
                seed_dict,
                node_dict.with_dupl,
                sampled_nbrs_per_node_dict,
                self._sampler.num_nodes,
                batch_dict.with_dupl,
                self.csc,
                self.disjoint,
            )

            node_dict.out = {
                ntype: torch.from_numpy(node_dict.out[ntype])
                for ntype in self._sampler.node_types
            }
            if self.disjoint:
                batch_dict.out = {
                    ntype: torch.from_numpy(batch_dict.out[ntype])
                    for ntype in self._sampler.node_types
                }

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
        else:
            src = seed
            node = src
            batch = src_batch if self.disjoint else None

            node_with_dupl = [torch.empty(0, dtype=torch.int64)]
            batch_with_dupl = [torch.empty(0, dtype=torch.int64)]
            edge = [torch.empty(0, dtype=torch.int64)]

            sampled_nbrs_per_node = []
            num_sampled_nodes = [seed.numel()]
            num_sampled_edges = []

            # loop over the layers
            for i, one_hop_num in enumerate(self.num_neighbors):
                out = await self.sample_one_hop(src, one_hop_num, seed_time,
                                                src_batch)
                if out.node.numel() == 0:
                    # no neighbors were sampled
                    break

                # remove duplicates
                src, node, src_batch, batch = remove_duplicates(
                    out, node, batch, self.disjoint)

                node_with_dupl.append(out.node)
                edge.append(out.edge)

                if self.disjoint:
                    batch_with_dupl.append(out.batch)

                if seed_time is not None and i < self.num_hops - 1:
                    # Get the seed time for the next layer based on the
                    # previous seed_time and sampled neighbors per node info:
                    seed_time = torch.repeat_interleave(
                        seed_time, torch.as_tensor(out.metadata[0]))

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

        if self.subgraph_type == SubgraphType.bidirectional:
            sampler_output = sampler_output.to_bidirectional()

        return sampler_output

    def get_sampler_output(
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

    def merge_sampler_outputs(
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
        :obj:`pyg-lib` :obj:`merge_sampler_outputs` function.

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
        r"""Sample one-hop neighbors for a :obj:`srcs`. If src node is located
        on a local partition, evaluates the :obj:`_sample_one_hop` function on
        a current machine. If src node is from a remote partition, send a
        request to a remote machine that contains this partition.

        Returns merged samplers outputs from local / remote machines.
        """
        partition_ids = self.graph_store.get_partition_ids_from_nids(srcs)
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
                           if seed_time is not None else None)

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
            return self.get_sampler_output(p_outputs, len(srcs),
                                           partition_ids[0], src_batch)

        return self.merge_sampler_outputs(partition_ids, partition_orders,
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
        else:
            rel_type = '__'.join(edge_type)
            colptr = self._sampler.colptr_dict[rel_type]
            row = self._sampler.row_dict[rel_type]
            node_time = self.node_time.get(edge_type[2],
                                           None) if self.node_time else None

        out = torch.ops.pyg.dist_neighbor_sample(
            colptr,
            row,
            input_nodes.to(colptr.dtype),
            num_neighbors,
            node_time,
            None,  # edge_time
            seed_time,
            None,  # TODO: edge_weight
            True,  # csc
            self.replace,
            self.subgraph_type != SubgraphType.induced,
            self.disjoint and node_time is not None,
            self.temporal_strategy,
        )
        node, edge, cumsum_neighbors_per_node = out

        if self.disjoint and node_time is not None:
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
            nlabels = {}
            nfeats = {}
            efeats = {}
            # Collect node labels of input node type.
            node_labels = self.feature_store.labels
            if node_labels is not None:
                nlabels = node_labels[output.node[self.input_type]]
            else:
                nlabels = None
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

        else:  # Homo
            # Collect node labels.
            nlabels = (self.feature_store.labels[output.node] if
                       (self.feature_store.labels is not None) else None)
            # Collect node features.
            if output.node is not None:
                fut = self.feature_store.lookup_features(
                    is_node_feat=True, index=output.node)
                nfeats = await to_asyncio_future(fut)
                nfeats = nfeats.to(torch.device("cpu"))
            # else:
            efeats = None
            # Collect edge features.
            if output.edge is not None and self.with_edge_attr:
                fut = self.feature_store.lookup_features(
                    is_node_feat=False, index=output.edge)
                efeats = await to_asyncio_future(fut)
                efeats = efeats.to(torch.device("cpu"))
            else:
                efeats = None

        output.metadata = (*output.metadata, nfeats, nlabels, efeats)
        return output

    @property
    def edge_permutation(self) -> None:
        return None

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(pid={mp.current_process().pid})'


# Sampling Utilities ##########################################################


def close_sampler(worker_id: int, sampler: DistNeighborSampler):
    # Make sure that mp.Queue is empty at exit and RAM is cleared:
    try:
        logging.info(f"Closing event loop for worker ID {worker_id}")
        sampler.event_loop.shutdown_loop()
    except AttributeError:
        pass
    logging.info(f"Closing RPC for worker ID {worker_id}")
    shutdown_rpc(graceful=True)
