from typing import Callable, Dict, List, Optional, Tuple, Union

import torch

from torch_geometric.distributed import (
    DistLoader,
    DistNeighborSampler,
    LocalFeatureStore,
    LocalGraphStore,
)
from torch_geometric.distributed.dist_context import DistContext, DistRole
from torch_geometric.loader import NodeLoader
from torch_geometric.sampler.base import SubgraphType
from torch_geometric.typing import EdgeType, InputNodes, OptTensor


class DistNeighborLoader(NodeLoader, DistLoader):
    r"""A distributed loader that preforms sampling from nodes.

    Args:
        data: A (:class:`~torch_geometric.data.FeatureStore`,
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
        rpc_worker_names (Dict[DistRole, List[str]]): RPC worker identifiers.
        concurrency (int, optional): RPC concurrency used for defining the
            maximum size of the asynchronous processing queue.
            (default: :obj:`1`)

        All other arguments follow the interface of
        :class:`torch_geometric.loader.NeighborLoader`.
    """
    def __init__(
        self,
        data: Tuple[LocalFeatureStore, LocalGraphStore],
        num_neighbors: Union[List[int], Dict[EdgeType, List[int]]],
        master_addr: str,
        master_port: Union[int, str],
        current_ctx: DistContext,
        rpc_worker_names: Dict[DistRole, List[str]],
        input_nodes: InputNodes = None,
        input_time: OptTensor = None,
        neighbor_sampler: Optional[DistNeighborSampler] = None,
        replace: bool = False,
        subgraph_type: Union[SubgraphType, str] = "directional",
        disjoint: bool = False,
        temporal_strategy: str = "uniform",
        time_attr: Optional[str] = None,
        transform: Optional[Callable] = None,
        concurrency: int = 1,
        filter_per_worker: Optional[bool] = False,
        async_sampling: bool = True,
        device: Optional[torch.device] = None,
        **kwargs,
    ):
        assert isinstance(data[0], LocalFeatureStore)
        assert isinstance(data[1], LocalGraphStore)
        assert concurrency >= 1, "RPC concurrency must be greater than 1"

        if input_time is not None and time_attr is None:
            raise ValueError("Received conflicting 'input_time' and "
                             "'time_attr' arguments: 'input_time' is set "
                             "while 'time_attr' is not set.")

        channel = torch.multiprocessing.Queue() if async_sampling else None

        if neighbor_sampler is None:
            neighbor_sampler = DistNeighborSampler(
                data=data,
                current_ctx=current_ctx,
                rpc_worker_names=rpc_worker_names,
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

        self.neighbor_sampler = neighbor_sampler

        DistLoader.__init__(
            self,
            channel=channel,
            master_addr=master_addr,
            master_port=master_port,
            current_ctx=current_ctx,
            rpc_worker_names=rpc_worker_names,
            **kwargs,
        )
        NodeLoader.__init__(
            self,
            data=data,
            node_sampler=neighbor_sampler,
            input_nodes=input_nodes,
            input_time=input_time,
            transform=transform,
            filter_per_worker=filter_per_worker,
            transform_sampler_output=self.channel_get,
            worker_init_fn=self.worker_init_fn,
            **kwargs,
        )

    def __repr__(self) -> str:
        return DistLoader.__repr__(self)
