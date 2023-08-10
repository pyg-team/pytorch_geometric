import torch
import logging
from typing import Callable, Optional, Tuple, Dict, Union, List

from torch_geometric.data import Data, HeteroData, GraphStore, FeatureStore
from torch_geometric.sampler import HeteroSamplerOutput, SamplerOutput
from torch_geometric.typing import EdgeType, InputNodes, OptTensor, as_str
from torch_geometric.loader.node_loader import NodeLoader

from .dist_loader import DistLoader
from .dist_neighbor_sampler import DistNeighborSampler
from .local_graph_store import LocalGraphStore
from .local_feature_store import LocalFeatureStore
from .dist_context import DistContext, DistRole

from torch_geometric.sampler.base import SubgraphType
from torch_geometric.loader.utils import filter_custom_store


class DistNeighborLoader(NodeLoader, DistLoader):
    r""" A distributed loader that preform sampling from nodes.
    Args:

      """

    def __init__(self,
                 data: Tuple[LocalFeatureStore, LocalGraphStore],
                 num_neighbors: Union[List[int], Dict[EdgeType, List[int]]],
                 master_addr: str,
                 master_port: Union[int, str],
                 current_ctx: DistContext,
                 rpc_worker_names: Dict[DistRole, List[str]],
                 neighbor_sampler: Optional[DistNeighborSampler] = None,
                 input_nodes: InputNodes = None,
                 input_time: OptTensor = None,
                 replace: bool = False,
                 subgraph_type: Union[SubgraphType, str] = 'directional',
                 disjoint: bool = False,
                 temporal_strategy: str = 'uniform',
                 time_attr: Optional[str] = None,
                 transform: Optional[Callable] = None,
                 transform_sampler_output: Optional[Callable] = None,
                 is_sorted: bool = False,
                 directed: bool = True,  # Deprecated.
                 with_edge: bool = False,
                 concurrency: int = 0,
                 collect_features: bool = True,
                 filter_per_worker: Optional[bool] = False,
                 async_sampling: bool = True,
                 device: torch.device = torch.device('cpu'),
                 **kwargs,
                 ):

        assert (isinstance(data[0], FeatureStore) and (
            data[1], GraphStore)), "Data needs to be Tuple[LocalFeatureStore, LocalGraphStore]"

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
        NodeLoader.__init__(self,
                            data=data,
                            node_sampler=neighbor_sampler,
                            input_nodes=input_nodes,
                            input_time=input_time,
                            transform=transform,
                            transform_sampler_output=transform_sampler_output,
                            filter_per_worker=filter_per_worker,
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
        # TODO: Align dist_sampler metadata output with original pyg sampler, such that filter_fn() from the NodeLoader can be used
        if self.channel:
            out = self.channel.get()
            logging.debug(
                f'{repr(self)} retrieved Sampler result from PyG MSG channel')

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

        elif isinstance(out, HeteroSamplerOutput):
            # data: Tuple[FeatureStore, GraphStore]
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

            try:
                data[input_type].input_id = out.metadata['bs']
                data[input_type].seed_time = out.metadata['input_id']
                data[input_type].batch_size = out.metadata['seed_time']
            except KeyError:
                pass

        else:
            raise TypeError(f"'{self.__class__.__name__}'' found invalid "
                            f"type: '{type(out)}'")

        return data if self.transform is None else self.transform(data)

    def __repr__(self):
        return DistLoader.__repr__(self)
