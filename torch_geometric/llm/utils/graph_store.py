from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch import Tensor

from torch_geometric.data import FeatureStore
from torch_geometric.distributed import LocalGraphStore
from torch_geometric.sampler import (
    BidirectionalNeighborSampler,
    NodeSamplerInput,
    SamplerOutput,
)
from torch_geometric.utils import index_sort

# A representation of an edge index, following the possible formats:
#    * default: Tensor, size = [2, num_edges]
#    *     Tensor[0, :] == row, Tensor[1, :] == col
#    * COO: (row, col)
#    * CSC: (row, colptr)
#    * CSR: (rowptr, col)
_EdgeTensorType = Union[Tensor, Tuple[Tensor, Tensor]]


class NeighborSamplingRAGGraphStore(LocalGraphStore):
    """Neighbor sampling based graph-store to store & retrieve graph data."""
    def __init__(  # type: ignore[no-untyped-def]
        self,
        feature_store: Optional[FeatureStore] = None,
        **kwargs,
    ):
        """Initializes the graph store.
        Optional feature store and neighbor sampling settings.

        Args:
        feature_store (optional): The feature store to use.
            None if not yet registered.
        **kwargs (optional):
            Additional keyword arguments for neighbor sampling.
        """
        self.feature_store = feature_store
        self.sample_kwargs = kwargs
        self._sampler_is_initialized = False
        self._config: Dict[str, Any] = {}

        # to be set by the config
        self.num_neighbors = None
        super().__init__()

    @property
    def config(self) -> Dict[str, Any]:
        """Get the config for the feature store."""
        return self._config

    def _set_from_config(self, config: Dict[str, Any], attr_name: str) -> None:
        """Set an attribute from the config.

        Args:
            config (Dict[str, Any]): Config dictionary
            attr_name (str): Name of attribute to set

        Raises:
            ValueError: If required attribute not found in config
        """
        if attr_name not in config:
            raise ValueError(
                f"Required config parameter '{attr_name}' not found")
        setattr(self, attr_name, config[attr_name])

    @config.setter  # type: ignore
    def config(self, config: Dict[str, Any]) -> None:
        """Set the config for the feature store.

        Args:
            config (Dict[str, Any]):
                Config dictionary containing required parameters

        Raises:
            ValueError: If required parameters missing from config
        """
        self._set_from_config(config, "num_neighbors")
        if hasattr(self, 'sampler'):
            self.sampler.num_neighbors = (  # type: ignore[has-type]
                self.num_neighbors)

        self._config = config

    def _init_sampler(self) -> None:
        """Initializes neighbor sampler with the registered feature store."""
        if self.feature_store is None:
            raise AttributeError("Feature store not registered yet.")
        assert self.num_neighbors is not None, \
            "Please set num_neighbors through config"
        self.sampler = BidirectionalNeighborSampler(
            data=(self.feature_store, self), num_neighbors=self.num_neighbors,
            **self.sample_kwargs)
        self._sampler_is_initialized = True

    def register_feature_store(self, feature_store: FeatureStore) -> None:
        """Registers a feature store with the graph store.

        :param feature_store: The feature store to register.
        """
        self.feature_store = feature_store
        self._sampler_is_initialized = False

    def put_edge_id(  # type: ignore[no-untyped-def]
            self, edge_id: Tensor, *args, **kwargs) -> bool:
        """Stores an edge ID in the graph store.

        :param edge_id: The edge ID to store.
        :return: Whether the operation was successful.
        """
        ret = super().put_edge_id(edge_id.contiguous(), *args, **kwargs)
        self._sampler_is_initialized = False
        return ret

    @property
    def edge_index(self) -> _EdgeTensorType:
        """Gets the edge index of the graph.

        :return: The edge index as a tensor.
        """
        return self.get_edge_index(*self.edge_idx_args, **self.edge_idx_kwargs)

    def put_edge_index(  # type: ignore[no-untyped-def]
            self, edge_index: _EdgeTensorType, *args, **kwargs) -> bool:
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
    @edge_index.setter  # type: ignore
    def edge_index(self, edge_index: _EdgeTensorType) -> None:
        """Sets the edge index of the graph.

        :param edge_index: The edge index to set.
        """
        # correct since we make node list from triples
        if isinstance(edge_index, Tensor):
            num_nodes = int(edge_index.max()) + 1
        else:
            assert isinstance(edge_index, tuple) \
                and isinstance(edge_index[0], Tensor) \
                and isinstance(edge_index[1], Tensor), \
                "edge_index must be a Tensor of [2, num_edges] \
                or a tuple of Tensors, (row, col)."

            num_nodes = int(edge_index[0].max()) + 1
        attr = dict(
            edge_type=None,
            layout='coo',
            size=(num_nodes, num_nodes),
            is_sorted=False,
        )
        # edge index needs to be sorted here and the perm saved for later
        col_sorted, self.perm = index_sort(edge_index[1], num_nodes,
                                           stable=True)
        row_sorted = edge_index[0][self.perm]
        edge_index_sorted = torch.stack([row_sorted, col_sorted], dim=0)
        self.put_edge_index(edge_index_sorted, **attr)

    def sample_subgraph(
        self,
        seed_nodes: Tensor,
    ) -> SamplerOutput:
        """Sample the graph starting from the given nodes using the
        in-built NeighborSampler.

        Args:
            seed_nodes (InputNodes): Seed nodes to start sampling from.
            num_neighbors (Optional[NumNeighborsType], optional): Parameters
                to determine how many hops and number of neighbors per hop.
                Defaults to None.

        Returns:
            Union[SamplerOutput, HeteroSamplerOutput]: NeighborSamplerOutput
                for the input.
        """
        # TODO add support for Hetero
        if not self._sampler_is_initialized:
            self._init_sampler()

        seed_nodes = seed_nodes.unique().contiguous()
        node_sample_input = NodeSamplerInput(input_id=None, node=seed_nodes)
        out = self.sampler.sample_from_nodes(  # type: ignore[has-type]
            node_sample_input)

        # edge ids need to be remapped to the original indices
        out.edge = self.perm[out.edge]

        return out
