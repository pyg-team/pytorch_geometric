import logging
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.multiprocessing as mp

from torch_geometric.distributed import LocalFeatureStore, LocalGraphStore
from torch_geometric.distributed.dist_context import DistContext, DistRole
from torch_geometric.distributed.event_loop import ConcurrentEventLoop
from torch_geometric.distributed.rpc import (
    RPCCallBase,
    RPCRouter,
    rpc_partition_to_workers,
    rpc_register,
    shutdown_rpc,
)
from torch_geometric.sampler import NeighborSampler
from torch_geometric.sampler.base import NumNeighbors, SubgraphType
from torch_geometric.typing import Dict, NumNeighbors, Tuple


class RpcSamplingCallee(RPCCallBase):
    r""" A wrapper for rpc callee that will perform rpc sampling from
    remote processes.
    """
    def __init__(self, sampler: NeighborSampler, device: torch.device):
        super().__init__()
        self.sampler = sampler
        self.device = device

    def rpc_async(self, *args, **kwargs):
        output = self.sampler._sample_one_hop(*args, **kwargs)

        return output

    def rpc_sync(self, *args, **kwargs):
        pass


class DistNeighborSampler:
    r"""An implementation of a distributed and asynchronised neighbor sampler
    used by :class:`~torch_geometric.distributed.DistNeighborLoader`."""
    def __init__(
        self,
        current_ctx: DistContext,
        rpc_worker_names: Dict[DistRole, List[str]],
        data: Tuple[LocalGraphStore, LocalFeatureStore],
        channel: mp.Queue(),
        num_neighbors: Optional[NumNeighbors] = None,
        with_edge: bool = True,
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

        self.data = data
        self.dist_graph = data[1]
        self.dist_feature = data[0]
        assert isinstance(
            self.dist_graph, LocalGraphStore
        ), "Provided data is in incorrect format: self.dist_graph must be "
        f"`LocalGraphStore`, got {type(self.dist_graph)}"
        assert isinstance(
            self.dist_feature, LocalFeatureStore
        ), "Provided data is in incorrect format: self.dist_feature must be "
        f"`LocalFeatureStore`, got {type(self.dist_feature)}"
        self.is_hetero = self.dist_graph.meta['is_hetero']

        self.num_neighbors = num_neighbors
        self.with_edge = with_edge
        self.channel = channel
        self.concurrency = concurrency
        self.device = device
        self.event_loop = None
        self.replace = replace
        self.subgraph_type = subgraph_type
        self.disjoint = disjoint
        self.temporal_strategy = temporal_strategy
        self.time_attr = time_attr
        self.csc = True  # always true?
        self.with_edge_attr = self.dist_feature.has_edge_attr()
        self.edge_permutation = None  # TODO: Debug edge_perm for LinkLoader

    def register_sampler_rpc(self) -> None:

        partition2workers = rpc_partition_to_workers(
            current_ctx=self.current_ctx,
            num_partitions=self.dist_graph.num_partitions,
            current_partition_idx=self.dist_graph.partition_idx)
        self.rpc_router = RPCRouter(partition2workers)
        self.dist_feature.set_rpc_router(self.rpc_router)

        self._sampler = NeighborSampler(
            data=(self.dist_feature, self.dist_graph),
            num_neighbors=self.num_neighbors,
            subgraph_type=self.subgraph_type,
            replace=self.replace,
            disjoint=self.disjoint,
            temporal_strategy=self.temporal_strategy,
            time_attr=self.time_attr,
        )
        rpc_sample_callee = RpcSamplingCallee(self._sampler, self.device)
        self.rpc_sample_callee_id = rpc_register(rpc_sample_callee)

    def init_event_loop(self) -> None:
        self.event_loop = ConcurrentEventLoop(self.concurrency)
        self.event_loop.start_loop()


# Sampling Utilities ##########################################################


def close_sampler(worker_id, sampler):
    # Make sure that mp.Queue is empty at exit and RAM is cleared
    try:
        logging.info(f"Closing event_loop in {sampler} worker-id {worker_id}")
        sampler.event_loop.shutdown_loop()
    except AttributeError:
        pass
    logging.info(f"Closing rpc in {sampler} worker-id {worker_id}")
    shutdown_rpc(graceful=True)
