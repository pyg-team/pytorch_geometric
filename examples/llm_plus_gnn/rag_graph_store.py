from typing import Optional, Union

import torch
from torch import Tensor

from torch_geometric.data import FeatureStore
from torch_geometric.distributed import LocalGraphStore
from torch_geometric.sampler import (
    HeteroSamplerOutput,
    NeighborSampler,
    NodeSamplerInput,
    SamplerOutput,
)
from torch_geometric.sampler.neighbor_sampler import NumNeighborsType
from torch_geometric.typing import EdgeTensorType, InputEdges, InputNodes


class NeighborSamplingRAGGraphStore(LocalGraphStore):
    def __init__(self, feature_store: Optional[FeatureStore] = None,
                 num_neighbors: NumNeighborsType = [100, 100, 100, 100,
                                                    100], **kwargs):
        self.feature_store = feature_store
        self.num_neighbors = num_neighbors
        self.sample_kwargs = kwargs
        self._sampler_is_initialized = False
        super().__init__()

    def _init_sampler(self):
        if self.feature_store is None:
            raise AttributeError("Feature store not registered yet.")
        self.sampler = NeighborSampler(data=(self.feature_store, self),
                                       num_neighbors=self.num_neighbors,
                                       **self.sample_kwargs)
        self._sampler_is_initialized = True

    def register_feature_store(self, feature_store: FeatureStore):
        self.feature_store = feature_store

    def put_edge_id(self, edge_id: Tensor, *args, **kwargs) -> bool:
        ret = super().put_edge_id(edge_id.contiguous(), *args, **kwargs)
        self._sampler_is_initialized = False
        return ret

    @property
    def edge_index(self):
        return self.get_edge_index(*self.edge_idx_args, **self.edge_idx_kwargs)

    def put_edge_index(self, edge_index: EdgeTensorType, *args,
                       **kwargs) -> bool:
        ret = super().put_edge_index(edge_index, *args, **kwargs)
        # HACK
        self.edge_idx_args = args
        self.edge_idx_kwargs = kwargs
        self._sampler_is_initialized = False
        return ret

    def sample_subgraph(self, seed_nodes: InputNodes, seed_edges: InputEdges,
                        *args,
                        **kwargs) -> Union[SamplerOutput, HeteroSamplerOutput]:
        if not self._sampler_is_initialized:
            self._init_sampler()

        # FIXME: Right now, only input nodes/edges as tensors are be supported
        if not isinstance(seed_nodes, Tensor):
            raise NotImplementedError
        if not isinstance(seed_edges, Tensor):
            raise NotImplementedError
        device = seed_nodes.device

        # TODO: Call sample_from_edges for seed_edges
        seed_edges = self.edge_index.to(device).T[seed_edges.to(
            device)].reshape((-1))
        seed_nodes = torch.cat((seed_nodes, seed_edges), dim=0)

        seed_nodes = seed_nodes.unique().contiguous()
        '''
        loader = NeighborLoader(data=(self.feature_store, self),
                                num_neighbors=[10], input_nodes=seed_nodes,
                                batch_size=num_nodes)
        loader = NodeLoader(data=(self.feature_store, self),
                            node_sampler=self.sampler, input_nodes=seed_nodes,
                            batch_size=num_nodes)
        '''
        node_sample_input = NodeSamplerInput(input_id=None, node=seed_nodes)
        out = self.sampler.sample_from_nodes(node_sample_input)

        return out
