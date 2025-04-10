from typing import Optional, Union

from torch import Tensor

from torch_geometric.data import FeatureStore
from torch_geometric.distributed import LocalGraphStore
from torch_geometric.sampler import (
    BidirectionalNeighborSampler,
    HeteroSamplerOutput,
    NodeSamplerInput,
    SamplerOutput,
)
from torch_geometric.sampler.neighbor_sampler import NumNeighborsType
from torch_geometric.typing import EdgeTensorType, InputEdges, InputNodes


class NeighborSamplingRAGGraphStore(LocalGraphStore):
    """A graph store that uses neighbor sampling to store and retrieve graph data.
    """
    def __init__(self, feature_store: Optional[FeatureStore] = None,
                 num_neighbors: NumNeighborsType = [1], **kwargs):
        """Initializes the graph store with an optional feature store and neighbor sampling settings.

        :param feature_store: The feature store to use, or None if not yet registered.
        :param num_neighbors: The number of neighbors to sample for each node.
        :param kwargs: Additional keyword arguments for neighbor sampling.
        """
        self.feature_store = feature_store
        self._num_neighbors = num_neighbors
        self.sample_kwargs = kwargs
        self._sampler_is_initialized = False
        super().__init__()

    def _init_sampler(self):
        """Initializes the neighbor sampler with the registered feature store.
        """
        if self.feature_store is None:
            raise AttributeError("Feature store not registered yet.")
        self.sampler = BidirectionalNeighborSampler(
            data=(self.feature_store, self), num_neighbors=self._num_neighbors,
            **self.sample_kwargs)
        self._sampler_is_initialized = True

    def register_feature_store(self, feature_store: FeatureStore):
        """Registers a feature store with the graph store.

        :param feature_store: The feature store to register.
        """
        self.feature_store = feature_store
        self._sampler_is_initialized = False

    def put_edge_id(self, edge_id: Tensor, *args, **kwargs) -> bool:
        """Stores an edge ID in the graph store.

        :param edge_id: The edge ID to store.
        :return: Whether the operation was successful.
        """
        ret = super().put_edge_id(edge_id.contiguous(), *args, **kwargs)
        self._sampler_is_initialized = False
        return ret

    @property
    def edge_index(self):
        """Gets the edge index of the graph.

        :return: The edge index as a tensor.
        """
        return self.get_edge_index(*self.edge_idx_args, **self.edge_idx_kwargs)

    def put_edge_index(self, edge_index: EdgeTensorType, *args,
                       **kwargs) -> bool:
        """Stores an edge index in the graph store.

        :param edge_index: The edge index to store.
        :return: Whether the operation was successful.
        """
        ret = super().put_edge_index(edge_index, *args, **kwargs)
        # HACK
        self.edge_idx_args = args
        self.edge_idx_kwargs = kwargs
        self._sampler_is_initialized = False
        return ret

    # HACKY
    @edge_index.setter
    def edge_index(self, edge_index: EdgeTensorType):
        """Sets the edge index of the graph.

        :param edge_index: The edge index to set.
        """
        # correct since we make node list from triples
        num_nodes = edge_index.max() + 1
        attr = dict(
            edge_type=None,
            layout='coo',
            size=(num_nodes, num_nodes),
            is_sorted=True,
        )
        self.put_edge_index(edge_index, **attr)

    @property
    def num_neighbors(self):
        """Gets the number of neighbors to sample for each node.

        :return: The number of neighbors.
        """
        return self._num_neighbors

    @num_neighbors.setter
    def num_neighbors(self, num_neighbors: NumNeighborsType):
        """Sets the number of neighbors to sample for each node.

        :param num_neighbors: The new number of neighbors.
        """
        self._num_neighbors = num_neighbors
        if hasattr(self, 'sampler'):
            self.sampler.num_neighbors = num_neighbors

    def sample_subgraph(
        self, seed_nodes: InputNodes, seed_edges: Optional[InputEdges] = None,
        num_neighbors: Optional[NumNeighborsType] = None
    ) -> Union[SamplerOutput, HeteroSamplerOutput]:
        """Sample the graph starting from the given nodes and edges using the
        in-built NeighborSampler.

        Args:
            seed_nodes (InputNodes): Seed nodes to start sampling from.
            seed_edges (Optional[InputEdges], optional): Seed edges to start sampling from.
                Defaults to None.
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
        if seed_edges:
            print("Note: seed edges currently unused")
            # if not isinstance(seed_edges, Tensor):
            #     raise NotImplementedError

        # TODO: Call sample_from_edges for seed_edges
        seed_nodes = seed_nodes.unique().contiguous()
        node_sample_input = NodeSamplerInput(input_id=None, node=seed_nodes)
        out = self.sampler.sample_from_nodes(node_sample_input)

        return out
