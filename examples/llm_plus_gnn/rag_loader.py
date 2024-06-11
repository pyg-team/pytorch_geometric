from torch_geometric.data import FeatureStore, GraphStore, Data, HeteroData
from typing import Protocol, Any, Tuple, Union, Optional, Callable
from torch_geometric.typing import EdgeTensorType, InputNodes, InputEdges
from abc import abstractmethod
from torch import Tensor
import torch


class RAGFeatureStore(Protocol, FeatureStore):
    """Feature store for remote GNN RAG backend."""

    @abstractmethod
    def retrieve_seed_nodes(self, query: Any, **kwargs) -> Tensor:
        """Makes a comparison between the query and all the nodes to get all the closest nodes. Return the indices of the nodes that are to be seeds for the RAG Sampler."""
        ...

    @abstractmethod
    def retieve_seed_edges(self, query: Any, **kwargs) -> EdgeTensorType:
        """Makes a comparison between the query and all the edges to get all the closest nodes. Returns the edge indices that are to be the seeds for the RAG Sampler."""
        ...


class RAGGraphStore(Protocol, GraphStore):
    """Graph store for remote GNN RAG backend."""

    @abstractmethod
    def retrieve_subgraph(self, seed_nodes: InputNodes, seed_edges: InputEdges) -> Union[Data, HeteroData]:
        """Sample a subgraph using the seeded nodes and edges."""
        ...

# TODO: Make compatible with Heterographs


class RagQueryLoader:
    def __init__(self, data: Tuple[RAGFeatureStore, RAGGraphStore], local_filter: Optional[Callable] = None, **kwargs):
        fstore, gstore = data
        self.feature_store: RAGFeatureStore = fstore
        self.graph_store: RAGGraphStore = gstore
        self.local_filter = local_filter
    
    def query(self, query: Any) -> Data:
        """Retrieve a subgraph associated with the query with all its feature attributes."""
        seed_nodes = self.feature_store.retrieve_seed_nodes(query)
        seed_edges = self.feature_store.retrieve_seed_edges(query)

        subgraph = self.graph_store.retrieve_subgraph(seed_nodes, seed_edges).edge_index

        nodes = torch.cat((subgraph[0], subgraph[1]), dim=0).unique()
        x = self.feature_store.multi_get_tensor(nodes)
        edge_attr = self.feature_store.multi_get_tensor(subgraph)

        data = Data(x=x, edge_attr=edge_attr, edge_index=subgraph)
        if self.local_filter:
            data = self.local_filter(data)
        return data
