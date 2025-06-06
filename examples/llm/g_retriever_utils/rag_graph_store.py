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
    def __init__(
        self,
        feature_store: Optional[FeatureStore] = None,
        num_neighbors: Optional[NumNeighborsType] = None,
        **kwargs,
    ) -> None:
        self.feature_store = feature_store
        self._num_neighbors = num_neighbors or [1]
        self.sample_kwargs = kwargs
        self._sampler_is_initialized = False
        super().__init__()

    def _init_sampler(self):
        if self.feature_store is None:
            raise AttributeError("Feature store not registered yet.")
        self.sampler = NeighborSampler(data=(self.feature_store, self),
                                       num_neighbors=self._num_neighbors,
                                       **self.sample_kwargs)
        self._sampler_is_initialized = True

    def register_feature_store(self, feature_store: FeatureStore):
        self.feature_store = feature_store
        self._sampler_is_initialized = False

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

    @property
    def num_neighbors(self):
        return self._num_neighbors

    @num_neighbors.setter
    def num_neighbors(self, num_neighbors: NumNeighborsType):
        self._num_neighbors = num_neighbors
        if hasattr(self, 'sampler'):
            self.sampler.num_neighbors = num_neighbors

    def sample_subgraph(
        self, seed_nodes: InputNodes, seed_edges: InputEdges,
        num_neighbors: Optional[NumNeighborsType] = None
    ) -> Union[SamplerOutput, HeteroSamplerOutput]:
        """Sample the graph starting from the given nodes and edges using the
        in-built NeighborSampler.

        Args:
            seed_nodes (InputNodes): Seed nodes to start sampling from.
            seed_edges (InputEdges): Seed edges to start sampling from.
            num_neighbors (Optional[NumNeighborsType], optional): Parameters
                to determine how many hops and number of neighbors per hop.
                Defaults to None.

        Returns:
            Union[SamplerOutput, HeteroSamplerOutput]: NeighborSamplerOutput
                for the input.
        """
        if not self._sampler_is_initialized:
            self._init_sampler()
        if num_neighbors is not None:
            self.num_neighbors = num_neighbors

        # FIXME: Right now, only input nodes/edges as tensors are be supported
        if not isinstance(seed_nodes, Tensor):
            raise NotImplementedError
        if not isinstance(seed_edges, Tensor):
            raise NotImplementedError
        device = seed_nodes.device

        # TODO: Call sample_from_edges for seed_edges
        # Turning them into nodes for now.
        seed_edges = self.edge_index.to(device).T[seed_edges.to(
            device)].reshape(-1)
        seed_nodes = torch.cat((seed_nodes, seed_edges), dim=0)

        seed_nodes = seed_nodes.unique().contiguous()
        node_sample_input = NodeSamplerInput(input_id=None, node=seed_nodes)
        out = self.sampler.sample_from_nodes(node_sample_input)

        return out
