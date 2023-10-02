import logging
from typing import Any, Dict, List, Optional, Tuple, Union

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
from torch_geometric.typing import EdgeType

NumNeighborsType = Union[NumNeighbors, List[int], Dict[EdgeType, List[int]]]


class RPCSamplingCallee(RPCCallBase):
    r"""A wrapper for RPC callee that will perform RPC sampling from remote
    processes."""
    def __init__(self, sampler: NeighborSampler):
        super().__init__()
        self.sampler = sampler

    def rpc_async(self, *args, **kwargs) -> Any:
        return self.sampler._sample_one_hop(*args, **kwargs)

    def rpc_sync(self, *args, **kwargs) -> Any:
        pass


class DistNeighborSampler:
    r"""An implementation of a distributed and asynchronised neighbor sampler
    used by :class:`~torch_geometric.distributed.DistNeighborLoader`."""
    def __init__(
        self,
        current_ctx: DistContext,
        rpc_worker_names: Dict[DistRole, List[str]],
        data: Tuple[LocalGraphStore, LocalFeatureStore],
        num_neighbors: NumNeighborsType,
        channel: Optional[mp.Queue] = None,
        replace: bool = False,
        subgraph_type: Union[SubgraphType, str] = 'directional',
        disjoint: bool = False,
        temporal_strategy: str = 'uniform',
        time_attr: Optional[str] = None,
        concurrency: int = 1,
        **kwargs,
    ):
        self.current_ctx = current_ctx
        self.rpc_worker_names = rpc_worker_names

        self.feature_store, self.graph_store = data
        assert isinstance(self.dist_graph, LocalGraphStore)
        assert isinstance(self.dist_feature_store, LocalFeatureStore)
        self.is_hetero = self.dist_graph.meta['is_hetero']

        self.num_neighbors = num_neighbors
        self.channel = channel or mp.Queue()
        self.concurrency = concurrency
        self.event_loop = None
        self.replace = replace
        self.subgraph_type = SubgraphType(subgraph_type)
        self.disjoint = disjoint
        self.temporal_strategy = temporal_strategy
        self.time_attr = time_attr
        self.with_edge_attr = self.dist_feature.has_edge_attr()
        self.edge_permutation = None  # TODO: Debug edge_perm for LinkLoader

    def register_sampler_rpc(self) -> None:
        partition2workers = rpc_partition_to_workers(
            current_ctx=self.current_ctx,
            num_partitions=self.dist_graph.num_partitions,
            current_partition_idx=self.dist_graph.partition_idx,
        )
        self.rpc_router = RPCRouter(partition2workers)
        self.dist_feature.set_rpc_router(self.rpc_router)

        self._sampler = NeighborSampler(
            data=(self.dist_feature_store, self.dist_graph_store),
            num_neighbors=self.num_neighbors,
            subgraph_type=self.subgraph_type,
            replace=self.replace,
            disjoint=self.disjoint,
            temporal_strategy=self.temporal_strategy,
            time_attr=self.time_attr,
        )
        rpc_sample_callee = RPCSamplingCallee(self._sampler)
        self.rpc_sample_callee_id = rpc_register(rpc_sample_callee)

    def init_event_loop(self) -> None:
        self.event_loop = ConcurrentEventLoop(self.concurrency)
        self.event_loop.start_loop()


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
