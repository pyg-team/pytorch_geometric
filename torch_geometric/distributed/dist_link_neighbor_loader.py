from typing import Optional

import torch
import logging 

from torch_geometric.sampler.base import (
    EdgeSamplerInput, SamplingType, SamplingConfig, NegativeSampling
)
from torch_geometric.typing import InputEdges, NumNeighbors
from typing import Callable, Optional, Tuple, Dict, Union, List
from .local_graph_store import LocalGraphStore
from .local_feature_store import LocalFeatureStore
from .dist_loader import DistLoader
from .dist_neighbor_sampler import DistNeighborSampler
from torch_geometric.sampler.base import SubgraphType
from .dist_context import DistContext, DistRole
from torch_geometric.loader.link_loader import LinkLoader
from torch_geometric.distributed.local_graph_store import LocalGraphStore
from torch_geometric.distributed.local_feature_store import LocalFeatureStore
from ..typing import Tuple, Dict, Union
from torch_geometric.data import Data, HeteroData, GraphStore, FeatureStore
from torch_geometric.sampler import HeteroSamplerOutput, SamplerOutput
from torch_geometric.typing import EdgeType, InputNodes, OptTensor, as_str
from torch_geometric.loader.utils import filter_custom_store

class DistLinkNeighborLoader(LinkLoader, DistLoader):
    r""" A distributed loader that preform sampling from edges.
    Args:
        data: A (:class:`~torch_geometric.data.FeatureStore`,
            :class:`~torch_geometric.data.GraphStore`) data object.
        num_neighbors (List[int] or Dict[Tuple[str, str, str], List[int]]): The
            number of neighbors to sample for each node in each iteration.
            If an entry is set to :obj:`-1`, all neighbors will be included.
            In heterogeneous graphs, may also take in a dictionary denoting
            the amount of neighbors to sample for each individual edge type.
        current_ctx (DistContext): Distributed context info of the current process.
        rpc_worker_names (Dict[DistRole, List[str]]): RPC workers identifiers.
        master_addr (str): RPC address for distributed loaders communication, 
            IP of the master node.
        master_port (Union[int, str]): Open port for RPC communication with 
            the master node.
        channel (mp.Queue): A communication channel for sample messages that 
            allows for asynchronous processing of the sampler calls.
            num_rpc_threads (Optional[int], optional): The number of threads in the
            thread-pool used by
            :class:`~torch.distributed.rpc.TensorPipeAgent` to execute
            requests (default: 16).
        rpc_timeout (Optional[int], optional): The default timeout, 
            in seconds, for RPC requests (default: 60 seconds). If the RPC has not
            completed in this timeframe, an exception indicating so will
            be raised. Callers can override this timeout for individual
            RPCs in :meth:`~torch.distributed.rpc.rpc_sync` and
            :meth:`~torch.distributed.rpc.rpc_async` if necessary. (default: 180)
        concurrency (Optional[int], optional): RPC concurrency used for defining 
            max size of asynchronous processing queue. 
        edge_label_index (Tensor or EdgeType or Tuple[EdgeType, Tensor]):
            The edge indices, holding source and destination nodes to start
            sampling from.
            If set to :obj:`None`, all edges will be considered.
            In heterogeneous graphs, needs to be passed as a tuple that holds
            the edge type and corresponding edge indices.
            (default: :obj:`None`)

        All other Args follow the input type of the standard torch_geometric.loader.LinkLoader.

        **kwargs (optional): Additional arguments of
            :class:`torch.utils.data.DataLoader`, such as :obj:`batch_size`,
            :obj:`shuffle`, :obj:`drop_last` or :obj:`num_workers`.
    """

    def __init__(self,
                 data: Tuple[LocalFeatureStore, LocalGraphStore],
                 num_neighbors: Union[List[int], Dict[EdgeType, List[int]]],
                 master_addr: str,
                 master_port: Union[int, str],
                 current_ctx: DistContext,
                 rpc_worker_names: Dict[DistRole, List[str]],
                 neighbor_sampler: Optional[DistNeighborSampler] = None,
                 edge_label_index: InputEdges = None,
                 edge_label: OptTensor = None,
                 edge_label_time: OptTensor = None,
                 replace: bool = False,
                 subgraph_type: Union[SubgraphType, str] = 'directional',
                 disjoint: bool = False,
                 temporal_strategy: str = 'uniform',
                 neg_sampling: Optional[NegativeSampling] = None,
                 neg_sampling_ratio: Optional[Union[int, float]] = None,
                 time_attr: Optional[str] = None,
                 transform: Optional[Callable] = None,
                 is_sorted: bool = False,
                 filter_per_worker: Optional[bool] = None,
                 directed: bool = True,  # Deprecated.
                 concurrency: int = 1,
                 async_sampling: bool = True,
                 device: Optional[torch.device] = torch.device('cpu'),
                 **kwargs
                 ):

        
        assert (isinstance(data[0], LocalFeatureStore) and (
            data[1], LocalGraphStore)), "Data needs to be Tuple[LocalFeatureStore, LocalGraphStore]"
        
        assert concurrency >= 1, "RPC concurrency must be greater than 1."
        
        channel = torch.multiprocessing.Queue() if async_sampling else None

        if (edge_label_time is not None) != (time_attr is not None):
            raise ValueError(
                f"Received conflicting 'edge_label_time' and 'time_attr' "
                f"arguments: 'edge_label_time' is "
                f"{'set' if edge_label_time is not None else 'not set'} "
                f"while 'time_attr' is "
                f"{'set' if time_attr is not None else 'not set'}. "
                f"Both arguments must be provided for temporal sampling."
            )

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
                is_sorted=is_sorted,
                share_memory=kwargs.get('num_workers', 0) > 0,
                directed=directed,
                device=device,
                channel=channel,
                concurrency=concurrency,
            )
            
        self.neighbor_sampler = neighbor_sampler

        DistLoader.__init__(self,
                            channel=channel,
                            master_addr=master_addr,
                            master_port=master_port,
                            current_ctx=current_ctx,
                            rpc_worker_names=rpc_worker_names,
                            **kwargs
                            )
        LinkLoader.__init__(self,
                            data=data,
                            link_sampler=neighbor_sampler,
                            edge_label_index=edge_label_index,
                            edge_label=edge_label,
                            neg_sampling=neg_sampling,
                            neg_sampling_ratio=neg_sampling_ratio,
                            transform=transform,
                            filter_per_worker=filter_per_worker,
                            worker_init_fn=self.worker_init_fn,
                            transform_sampler_output=self.channel_get,
                            **kwargs
                            )
      
    def __repr__(self):
      return DistLoader.__repr__(self)