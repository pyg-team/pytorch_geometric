from typing import Callable, Dict, List, Optional, Tuple, Union

import torch

from torch_geometric.distributed.dist_context import DistContext, DistRole
from torch_geometric.distributed.dist_loader import DistLoader
from torch_geometric.distributed.dist_neighbor_sampler import (
    DistNeighborSampler,
)
from torch_geometric.distributed.local_feature_store import LocalFeatureStore
from torch_geometric.distributed.local_graph_store import LocalGraphStore
from torch_geometric.loader.node_loader import NodeLoader
from torch_geometric.sampler.base import SubgraphType
from torch_geometric.typing import EdgeType, InputNodes, OptTensor


class DistNeighborLoader(NodeLoader, DistLoader):
    r"""A distributed loader that preform sampling from nodes.
    Args:
        data: A (:class:`~torch_geometric.data.FeatureStore`,
            :class:`~torch_geometric.data.GraphStore`) data object.
        num_neighbors (List[int] or Dict[Tuple[str, str, str], List[int]]):
            The number of neighbors to sample for each node in each iteration.
            If an entry is set to :obj:`-1`, all neighbors will be included.
            In heterogeneous graphs, may also take in a dictionary denoting
            the amount of neighbors to sample for each individual edge type.
        current_ctx (DistContext): Distributed context info of the current
            process.
        rpc_worker_names (Dict[DistRole, List[str]]): RPC workers identifiers.
        master_addr (str): RPC address for distributed loaders communication,
            IP of the master node.
        master_port (Union[int, str]): Open port for RPC communication with
            the master node.
        channel (mp.Queue): A communication channel for sample messages that
            allows for asynchronous processing of the sampler calls.
            num_rpc_threads (Optional[int], optional): The number of threads
            in the thread-pool used by
            :class:`~torch.distributed.rpc.TensorPipeAgent` to execute
            requests (default: 16).
        rpc_timeout (Optional[int], optional): The default timeout,
            in seconds, for RPC requests (default: 60 seconds). If the RPC
            has not completed in this timeframe, an exception indicating so
            will be raised. Callers can override this timeout for individual
            RPCs in :meth:`~torch.distributed.rpc.rpc_sync` and
            :meth:`~torch.distributed.rpc.rpc_async` if necessary.
            (default: 180)
        concurrency (Optional[int], optional): RPC concurrency used for
            defining max size of asynchronous processing queue.
        input_nodes (torch.Tensor or str or Tuple[str, torch.Tensor]): The
            indices of nodes for which neighbors are sampled to create
            mini-batches.
            Needs to be either given as a :obj:`torch.LongTensor` or
            :obj:`torch.BoolTensor`.
            If set to :obj:`None`, all nodes will be considered.
            In heterogeneous graphs, needs to be passed as a tuple that holds
            the node type and node indices. (default: :obj:`None`)

        All other Args follow the input type of the standard
        torch_geometric.loader.NodeLoader.

        **kwargs (optional): Additional arguments of
            :class:`torch.utils.data.DataLoader`, such as :obj:`batch_size`,
            :obj:`shuffle`, :obj:`drop_last` or :obj:`num_workers`.
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
            is_sorted: bool = False,
            directed: bool = True,  # Deprecated.
            with_edge: bool = True,
            concurrency: int = 1,
            filter_per_worker: Optional[bool] = False,
            async_sampling: bool = True,
            device: torch.device = torch.device("cpu"),
            **kwargs,
    ):
        assert isinstance(data[0], LocalFeatureStore) and (
            data[1],
            LocalGraphStore,
        ), "Data needs to be Tuple[LocalFeatureStore, LocalGraphStore]"

        assert concurrency >= 1, "RPC concurrency must be greater than 1."

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
                with_edge=with_edge,
                replace=replace,
                subgraph_type=subgraph_type,
                disjoint=disjoint,
                temporal_strategy=temporal_strategy,
                time_attr=time_attr,
                is_sorted=is_sorted,
                share_memory=kwargs.get("num_workers", 0) > 0,
                directed=directed,
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

    def __repr__(self):
        return DistLoader.__repr__(self)
