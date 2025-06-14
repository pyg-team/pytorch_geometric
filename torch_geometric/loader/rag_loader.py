from abc import abstractmethod
from typing import Any, Callable, Dict, Optional, Protocol, Tuple, Union

import torch

from torch_geometric.data import Data, FeatureStore, HeteroData
from torch_geometric.sampler import HeteroSamplerOutput, SamplerOutput
from torch_geometric.typing import InputEdges, InputNodes
from torch_geometric.utils.rag.vectorrag import VectorRetriever


class RAGFeatureStore(Protocol):
    """Feature store template for remote GNN RAG backend."""
    @abstractmethod
    def retrieve_seed_nodes(self, query: Any, **kwargs) -> InputNodes:
        """Makes a comparison between the query and all the nodes to get all
        the closest nodes. Return the indices of the nodes that are to be seeds
        for the RAG Sampler.
        """
        ...
    
    @property
    @abstractmethod
    def config(self) -> Dict[str, Any]:
        """Get the config for the RAGFeatureStore."""
        ...
    
    @config.setter
    @abstractmethod
    def config(self, config: Dict[str, Any]):
        """Set the config for the RAGFeatureStore."""
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

    @property
    @abstractmethod
    def config(self) -> Dict[str, Any]:
        """Get the config for the RAGGraphStore."""
        ...

    @config.setter
    @abstractmethod
    def config(self, config: Dict[str, Any]):
        """Set the config for the RAGGraphStore."""
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
    def __init__(self, graph_data: Tuple[RAGFeatureStore, RAGGraphStore],
                 subgraph_filter: Optional[Callable[[Data, Any], Data]] = None,
                 augment_query: bool = False,
                 vector_retriever: Optional[VectorRetriever] = None,
                 config: Optional[Dict[str, Any]] = None):
        """Loader meant for making queries from a remote backend.

        Args:
            graph_data (Tuple[RAGFeatureStore, RAGGraphStore]): Remote FeatureStore
                and GraphStore to load from. Assumed to conform to the
                protocols listed above.
            subgraph_filter (Optional[Callable[[Data, Any], Data]], optional):
                Optional local transform to apply to data after retrieval.
                Defaults to None.
            augment_query (bool, optional): Whether to augment the query with
                retrieved documents. Defaults to False.
            vector_retriever (Optional[VectorRetriever], optional): VectorRetriever to use for
                retrieving documents. Defaults to None.
            config (Optional[Dict[str, Any]], optional): Config to pass into
                the RAGQueryLoader. Defaults to None.
        """
        fstore, gstore = graph_data
        self.vector_retriever = vector_retriever
        self.augment_query = augment_query
        self.feature_store = fstore
        self.graph_store = gstore
        self.graph_store.edge_index = self.graph_store.edge_index.contiguous()
        self.graph_store.register_feature_store(self.feature_store)
        self.subgraph_filter = subgraph_filter
        self.config = config

    def _propagate_config(self, config: Dict[str, Any]):
        """Propagate the config the relevant components.
        """
        self.feature_store.config = config
        self.graph_store.config = config

    @property
    def config(self):
        """Get the config for the RAGQueryLoader.
        """
        return self._config

    @config.setter
    def config(self, config: Dict[str, Any]):
        """Set the config for the RAGQueryLoader.

        Args:
            config (Dict[str, Any]): The config to set.
        """
        self._propagate_config(config)
        self._config = config

    def query(self, query: Any) -> Data:
        """Retrieve a subgraph associated with the query with all its feature
        attributes.
        """
        if self.vector_retriever:
            retrieved_docs = self.vector_retriever.query(query)

        if self.augment_query:
            query = [query] + retrieved_docs

        seed_nodes, query_enc = self.feature_store.retrieve_seed_nodes(query)

        subgraph_sample = self.graph_store.sample_subgraph(seed_nodes)
        node_idx, edge_idx = subgraph_sample.node, subgraph_sample.edge

        data = self.feature_store.load_subgraph(sample=subgraph_sample)
        # Useful for tracking what subset of the graph was sampled
        data.node_idx = node_idx
        data.edge_idx = edge_idx

        '''
        # need to use sampled edge_idx to index into original graph then reindex
        total_e_idx_t = self.graph_store.edge_index[:, data.edge_idx].t()
        data.node_idx = torch.tensor(
            list(
                dict.fromkeys(seed_nodes.tolist() +
                              total_e_idx_t.reshape(-1).tolist())))
        data.num_nodes = len(data.node_idx)

        # use node idx to get data.x
        data.x = self.feature_store.x[data.node_idx]
        list_edge_index = []

        # remap the edge_index
        remap_dict = {
            int(data.node_idx[i]): i
            for i in range(len(data.node_idx))
        }
        for src, dst in total_e_idx_t.tolist():
            list_edge_index.append(
                (remap_dict[int(src)], remap_dict[int(dst)]))
        data.edge_index = torch.tensor(list_edge_index).t()
        '''

        # apply local filter
        if self.subgraph_filter:
            data = self.subgraph_filter(data, query)
        if self.vector_retriever:
            data.text_context = retrieved_docs
        return data
