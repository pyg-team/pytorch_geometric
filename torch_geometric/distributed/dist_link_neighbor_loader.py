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
    """

    def __init__(self,
                 data: Tuple[LocalFeatureStore, LocalGraphStore],
                 num_neighbors: Union[List[int], Dict[EdgeType, List[int]]],
                 master_addr: str,
                 master_port: Union[int, str],
                 current_ctx: DistContext,
                 rpc_worker_names: Dict[DistRole, List[str]],
                 neighbor_sampler: Optional[DistNeighborSampler] = None,
                 with_edge: bool = True,
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
                 transform_sampler_output: Optional[Callable] = None,
                 is_sorted: bool = False,
                 filter_per_worker: Optional[bool] = None,
                 directed: bool = True,  # Deprecated.
                 concurrency: int = 4,
                 collect_features: bool = True,
                 async_sampling: bool = True,
                 device: Optional[torch.device] = torch.device('cpu'),
                 **kwargs
                 ):

        
        assert (isinstance(data[0], FeatureStore) and (data[1], GraphStore)), "Data needs to be Tuple[LocalFeatureStore, LocalGraphStore]"
        
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
                with_edge=with_edge,
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
                collect_features=collect_features
            )
        DistLoader.__init__(self,
                            neighbor_sampler=neighbor_sampler,
                            channel=channel,
                            master_addr=master_addr,
                            master_port=master_port,
                            current_ctx=current_ctx,
                            rpc_worker_names=rpc_worker_names,
                            **kwargs
                            )
        LinkLoader.__init__(self,
                            # Tuple[FeatureStore, GraphStore]
                            data=data,
                            link_sampler=neighbor_sampler,
                            edge_label_index=edge_label_index,
                            edge_label=edge_label,
                            neg_sampling=neg_sampling,
                            neg_sampling_ratio=neg_sampling_ratio,
                            transform=transform,
                            transform_sampler_output=transform_sampler_output,
                            filter_per_worker=filter_per_worker,
                            worker_init_fn=self.worker_init_fn,
                            **kwargs
                            )
        
    def filter_fn(
        self,
        out: Union[SamplerOutput, HeteroSamplerOutput],
        ) -> Union[Data, HeteroData]:
        r"""Joins the sampled nodes with their corresponding features,
        returning the resulting :class:`~torch_geometric.data.Data` or
        :class:`~torch_geometric.data.HeteroData` object to be used downstream.
        """
        # TODO: Align dist_sampler metadata output with original pyg sampler, such that filter_fn() from the LinkLoader can be used
        if self.channel:
          out = self.channel.get()
          logging.debug(f'{repr(self)} retrieved Sampler result from PyG MSG channel')
          
        if isinstance(out, SamplerOutput):
          edge_index = torch.stack([out.row, out.col])
          data = Data(x=out.metadata['nfeats'],
                      edge_index=edge_index,
                      edge_attr=out.metadata['efeats'],
                      y=out.metadata['nlabels']
                      )
          
          data.edge = out.edge
          data.node = out.node
          data.batch = out.batch
          data.num_sampled_nodes = out.num_sampled_nodes
          data.num_sampled_edges = out.num_sampled_edges
          
          try:
            data.batch_size = out.metadata['bs']
            data.input_id = out.metadata['input_id']
            data.seed_time = out.metadata['seed_time']
          except KeyError:
            pass

          if self.neg_sampling is None or self.neg_sampling.is_binary():
              # TODO
              pass
              # data.edge_label_index = out.metadata[1]
              # data.edge_label = out.metadata[2]
              # data.edge_label_time = out.metadata[3]
          elif self.neg_sampling.is_triplet():
              # TODO
              pass
              # data.src_index = out.metadata[1]
              # data.dst_pos_index = out.metadata[2]
              # data.dst_neg_index = out.metadata[3]
              # data.seed_time = out.metadata[4]
              # # Sanity removals in case `edge_label_index` and
              # # `edge_label_time` are attributes of the base `data` object:
              # del data.edge_label_index  # Sanity removals.
              # del data.edge_label_time

        elif isinstance(out, HeteroSamplerOutput):

            data = filter_custom_store(*self.data, out.node, out.row,
                                        out.col, out.edge, self.custom_cls)

            for key, node in out.node.items():
                if 'n_id' not in data[key]:
                    data[key].n_id = node

            for key, edge in (out.edge or {}).items():
                if 'e_id' not in data[key]:
                    data[key].e_id = edge

            data.set_value_dict('batch', out.batch)
            data.set_value_dict('num_sampled_nodes', out.num_sampled_nodes)
            data.set_value_dict('num_sampled_edges', out.num_sampled_edges)

            input_type = self.input_data.input_type
            data[input_type].input_id = out.metadata[0]

            if self.neg_sampling is None or self.neg_sampling.is_binary():
                data[input_type].edge_label_index = out.metadata[1]
                data[input_type].edge_label = out.metadata[2]
                data[input_type].edge_label_time = out.metadata[3]
            elif self.neg_sampling.is_triplet():
                data[input_type[0]].src_index = out.metadata[1]
                data[input_type[-1]].dst_pos_index = out.metadata[2]
                data[input_type[-1]].dst_neg_index = out.metadata[3]
                data[input_type[0]].seed_time = out.metadata[4]
                data[input_type[-1]].seed_time = out.metadata[4]
                # Sanity removals in case `edge_label_index` and
                # `edge_label_time` are attributes of the base `data` object:
                if input_type in data.edge_types:
                    del data[input_type].edge_label_index
                    del data[input_type].edge_label_time

        else:
            raise TypeError(f"'{self.__class__.__name__}'' found invalid "
                            f"type: '{type(out)}'")

        return data if self.transform is None else self.transform(data)
      
    def __repr__(self):
      return DistLoader.__repr__(self)