from abc import abstractmethod
from typing import Any, Callable, Dict, Optional, Protocol, Tuple, Union

from torch_geometric.data import Data, FeatureStore, HeteroData
from torch_geometric.sampler import HeteroSamplerOutput, SamplerOutput
from torch_geometric.typing import InputEdges, InputNodes


class RAGFeatureStore(Protocol):
    """Feature store template for remote GNN RAG backend."""
    @abstractmethod
    def retrieve_seed_nodes(self, query: Any, **kwargs) -> InputNodes:
        """Makes a comparison between the query and all the nodes to get all
        the closest nodes. Return the indices of the nodes that are to be seeds
        for the RAG Sampler.
        """
        ...

    @abstractmethod
    def retrieve_seed_edges(self, query: Any, **kwargs) -> InputEdges:
        """Makes a comparison between the query and all the edges to get all
        the closest nodes. Returns the edge indices that are to be the seeds
        for the RAG Sampler.
        """
        ...

    @abstractmethod
    def load_subgraph(
        self, sample: Union[SamplerOutput, HeteroSamplerOutput]
    ) -> Union[Data, HeteroData]:
        """Combines sampled subgraph output with features in a Data object."""
        ...


class RAGGraphStore(Protocol):
    """Graph store template for remote GNN RAG backend."""
    @abstractmethod
    def sample_subgraph(self, seed_nodes: InputNodes, seed_edges: InputEdges,
                        **kwargs) -> Union[SamplerOutput, HeteroSamplerOutput]:
        """Sample a subgraph using the seeded nodes and edges."""
        ...

    @abstractmethod
    def register_feature_store(self, feature_store: FeatureStore):
        """Register a feature store to be used with the sampler. Samplers need
        info from the feature store in order to work properly on HeteroGraphs.
        """
        ...


# TODO: Make compatible with Heterographs


class RAGQueryLoader:
    """Loader meant for making RAG queries from a remote backend."""
    def __init__(self, data: Tuple[RAGFeatureStore, RAGGraphStore],
                 local_filter: Optional[Callable[[Data, Any], Data]] = None,
                 seed_nodes_kwargs: Optional[Dict[str, Any]] = None,
                 seed_edges_kwargs: Optional[Dict[str, Any]] = None,
                 sampler_kwargs: Optional[Dict[str, Any]] = None,
                 loader_kwargs: Optional[Dict[str, Any]] = None):
        """Loader meant for making queries from a remote backend.

        Args:
            data (Tuple[RAGFeatureStore, RAGGraphStore]): Remote FeatureStore
                and GraphStore to load from. Assumed to conform to the
                protocols listed above.
            local_filter (Optional[Callable[[Data, Any], Data]], optional):
                Optional local transform to apply to data after retrieval.
                Defaults to None.
            seed_nodes_kwargs (Optional[Dict[str, Any]], optional): Paramaters
                to pass into process for fetching seed nodes. Defaults to None.
            seed_edges_kwargs (Optional[Dict[str, Any]], optional): Parameters
                to pass into process for fetching seed edges. Defaults to None.
            sampler_kwargs (Optional[Dict[str, Any]], optional): Parameters to
                pass into process for sampling graph. Defaults to None.
            loader_kwargs (Optional[Dict[str, Any]], optional): Parameters to
                pass into process for loading graph features. Defaults to None.
        """
        fstore, gstore = data
        self.feature_store = fstore
        self.graph_store = gstore
        self.graph_store.register_feature_store(self.feature_store)
        self.local_filter = local_filter
        self.seed_nodes_kwargs = seed_nodes_kwargs or {}
        self.seed_edges_kwargs = seed_edges_kwargs or {}
        self.sampler_kwargs = sampler_kwargs or {}
        self.loader_kwargs = loader_kwargs or {}

    def query(self, query: Any) -> Data:
        """Retrieve a subgraph associated with the query with all its feature
        attributes.
        """
        seed_nodes = self.feature_store.retrieve_seed_nodes(
            query, **self.seed_nodes_kwargs)
        seed_edges = self.feature_store.retrieve_seed_edges(
            query, **self.seed_edges_kwargs)

        subgraph_sample = self.graph_store.sample_subgraph(
            seed_nodes, seed_edges, **self.sampler_kwargs)

        data = self.feature_store.load_subgraph(sample=subgraph_sample,
                                                **self.loader_kwargs)

        if self.local_filter:
            data = self.local_filter(data, query)
        return data
