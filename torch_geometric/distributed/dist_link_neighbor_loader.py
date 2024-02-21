from typing import Callable, Dict, List, Optional, Tuple, Union

import torch

from torch_geometric.distributed import (
    DistContext,
    DistLoader,
    DistNeighborSampler,
    LocalFeatureStore,
    LocalGraphStore,
)
from torch_geometric.loader import LinkLoader
from torch_geometric.sampler.base import NegativeSampling, SubgraphType
from torch_geometric.typing import EdgeType, InputEdges, OptTensor


class DistLinkNeighborLoader(LinkLoader, DistLoader):
    r"""A distributed loader that performs sampling from edges.

    Args:
        data (tuple): A (:class:`~torch_geometric.data.FeatureStore`,
            :class:`~torch_geometric.data.GraphStore`) data object.
        num_neighbors (List[int] or Dict[Tuple[str, str, str], List[int]]):
            The number of neighbors to sample for each node in each iteration.
            If an entry is set to :obj:`-1`, all neighbors will be included.
            In heterogeneous graphs, may also take in a dictionary denoting
            the amount of neighbors to sample for each individual edge type.
        master_addr (str): RPC address for distributed loader communication,
            *i.e.* the IP address of the master node.
        master_port (Union[int, str]): Open port for RPC communication with
            the master node.
        current_ctx (DistContext): Distributed context information of the
            current process.
        concurrency (int, optional): RPC concurrency used for defining the
            maximum size of the asynchronous processing queue.
            (default: :obj:`1`)

    All other arguments follow the interface of
    :class:`torch_geometric.loader.LinkNeighborLoader`.
    """
    def __init__(
        self,
        data: Tuple[LocalFeatureStore, LocalGraphStore],
        num_neighbors: Union[List[int], Dict[EdgeType, List[int]]],
        master_addr: str,
        master_port: Union[int, str],
        current_ctx: DistContext,
        edge_label_index: InputEdges = None,
        edge_label: OptTensor = None,
        edge_label_time: OptTensor = None,
        dist_sampler: Optional[DistNeighborSampler] = None,
        replace: bool = False,
        subgraph_type: Union[SubgraphType, str] = "directional",
        disjoint: bool = False,
        temporal_strategy: str = "uniform",
        neg_sampling: Optional[NegativeSampling] = None,
        neg_sampling_ratio: Optional[Union[int, float]] = None,
        time_attr: Optional[str] = None,
        transform: Optional[Callable] = None,
        concurrency: int = 1,
        num_rpc_threads: int = 16,
        filter_per_worker: Optional[bool] = False,
        async_sampling: bool = True,
        device: Optional[torch.device] = None,
        **kwargs,
    ):
        assert isinstance(data[0], LocalFeatureStore)
        assert isinstance(data[1], LocalGraphStore)
        assert concurrency >= 1, "RPC concurrency must be greater than 1"

        if (edge_label_time is not None) != (time_attr is not None):
            raise ValueError(
                f"Received conflicting 'edge_label_time' and 'time_attr' "
                f"arguments: 'edge_label_time' is "
                f"{'set' if edge_label_time is not None else 'not set'} "
                f"while 'time_attr' is "
                f"{'set' if time_attr is not None else 'not set'}. "
                f"Both arguments must be provided for temporal sampling.")

        channel = torch.multiprocessing.Queue() if async_sampling else None

        if dist_sampler is None:
            dist_sampler = DistNeighborSampler(
                data=data,
                current_ctx=current_ctx,
                num_neighbors=num_neighbors,
                replace=replace,
                subgraph_type=subgraph_type,
                disjoint=disjoint,
                temporal_strategy=temporal_strategy,
                time_attr=time_attr,
                device=device,
                channel=channel,
                concurrency=concurrency,
            )

        DistLoader.__init__(
            self,
            channel=channel,
            master_addr=master_addr,
            master_port=master_port,
            current_ctx=current_ctx,
            dist_sampler=dist_sampler,
            num_rpc_threads=num_rpc_threads,
            **kwargs,
        )
        LinkLoader.__init__(
            self,
            data=data,
            link_sampler=dist_sampler,
            edge_label_index=edge_label_index,
            edge_label=edge_label,
            edge_label_time=edge_label_time,
            neg_sampling=neg_sampling,
            neg_sampling_ratio=neg_sampling_ratio,
            transform=transform,
            filter_per_worker=filter_per_worker,
            worker_init_fn=self.worker_init_fn,
            transform_sampler_output=self.channel_get if channel else None,
            **kwargs,
        )

    def __repr__(self) -> str:
        return DistLoader.__repr__(self)
