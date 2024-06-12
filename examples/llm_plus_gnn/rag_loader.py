from abc import abstractmethod
from typing import (
    Any,
    Callable,
    Iterable,
    Optional,
    Protocol,
    Tuple,
    Union,
    runtime_checkable,
)

import torch
from torch import Tensor

from torch_geometric.data import Data, FeatureStore, GraphStore, HeteroData
from torch_geometric.typing import InputEdges, InputNodes


@runtime_checkable
class RAGFeatureStore(Protocol):
    """Feature store for remote GNN RAG backend."""
    @abstractmethod
    def retrieve_seed_nodes(self, query: Iterable[Any],
                            **kwargs) -> Iterable[Tensor]:
        """Makes a comparison between the query and all the nodes to get all
        the closest nodes. Return the indices of the nodes that are to be seeds
        for the RAG Sampler.
        """
        ...

    @abstractmethod
    def retrieve_seed_edges(self, query: Iterable[Any],
                            **kwargs) -> Iterable[Tensor]:
        """Makes a comparison between the query and all the edges to get all
        the closest nodes. Returns the edge indices that are to be the seeds
        for the RAG Sampler.
        """
        ...


@runtime_checkable
class RAGGraphStore(Protocol):
    """Graph store for remote GNN RAG backend."""
    @abstractmethod
    def retrieve_subgraph(self, seed_nodes: InputNodes,
                          seed_edges: InputEdges) -> Union[Data, HeteroData]:
        """Sample a subgraph using the seeded nodes and edges."""
        ...

    @abstractmethod
    def register_feature_store(self, feature_store: FeatureStore):
        """Register a feature store to be used with the sampler."""
        ...


# TODO: Make compatible with Heterographs


class RagQueryLoader:
    def __init__(self, data: Tuple[FeatureStore, GraphStore],
                 local_filter: Optional[Callable] = None, **kwargs):
        fstore, gstore = data
        assert issubclass(type(fstore), RAGFeatureStore)
        assert issubclass(type(gstore), RAGGraphStore)
        self.feature_store: FeatureStore = fstore
        self.graph_store: GraphStore = gstore
        self.graph_store.register_feature_store(self.feature_store)
        self.local_filter = local_filter

    def query(self, query: Any) -> Data:
        """Retrieve a subgraph associated with the query with all its feature
        attributes.
        """
        seed_nodes = self.feature_store.retrieve_seed_nodes([query])[0]
        seed_edges = self.feature_store.retrieve_seed_edges([query])[0]

        subgraph = self.graph_store.retrieve_subgraph(seed_nodes,
                                                      seed_edges).edge_index

        nodes = torch.cat((subgraph[0], subgraph[1]), dim=0).unique()
        x = self.feature_store.multi_get_tensor(nodes)
        edge_attr = self.feature_store.multi_get_tensor(subgraph)

        data = Data(x=x, edge_attr=edge_attr, edge_index=subgraph)
        if self.local_filter:
            data = self.local_filter(data)
        return data
