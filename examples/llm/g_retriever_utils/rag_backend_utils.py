from dataclasses import dataclass
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Optional,
    Protocol,
    Tuple,
    Type,
    runtime_checkable,
)

import torch
from torch import Tensor
from torch.nn import Module

from torch_geometric.data import (
    FeatureStore,
    GraphStore,
    LargeGraphIndexer,
    TripletLike,
)
from torch_geometric.data.large_graph_indexer import EDGE_RELATION
from torch_geometric.distributed import (
    LocalFeatureStore,
    LocalGraphStore,
    Partitioner,
)
from torch_geometric.typing import EdgeType, NodeType

RemoteGraphBackend = Tuple[FeatureStore, GraphStore]

# TODO: Make everything compatible with Hetero graphs as well


# Adapted from LocalGraphStore
@runtime_checkable
class ConvertableGraphStore(Protocol):
    @classmethod
    def from_data(
        cls,
        edge_id: Tensor,
        edge_index: Tensor,
        num_nodes: int,
        is_sorted: bool = False,
    ) -> GraphStore:
        ...

    @classmethod
    def from_hetero_data(
        cls,
        edge_id_dict: Dict[EdgeType, Tensor],
        edge_index_dict: Dict[EdgeType, Tensor],
        num_nodes_dict: Dict[NodeType, int],
        is_sorted: bool = False,
    ) -> GraphStore:
        ...

    @classmethod
    def from_partition(cls, root: str, pid: int) -> GraphStore:
        ...


# Adapted from LocalFeatureStore
@runtime_checkable
class ConvertableFeatureStore(Protocol):
    @classmethod
    def from_data(
        cls,
        node_id: Tensor,
        x: Optional[Tensor] = None,
        y: Optional[Tensor] = None,
        edge_id: Optional[Tensor] = None,
        edge_attr: Optional[Tensor] = None,
    ) -> FeatureStore:
        ...

    @classmethod
    def from_hetero_data(
        cls,
        node_id_dict: Dict[NodeType, Tensor],
        x_dict: Optional[Dict[NodeType, Tensor]] = None,
        y_dict: Optional[Dict[NodeType, Tensor]] = None,
        edge_id_dict: Optional[Dict[EdgeType, Tensor]] = None,
        edge_attr_dict: Optional[Dict[EdgeType, Tensor]] = None,
    ) -> FeatureStore:
        ...

    @classmethod
    def from_partition(cls, root: str, pid: int) -> FeatureStore:
        ...


class RemoteDataType(Enum):
    DATA = auto()
    PARTITION = auto()


@dataclass
class RemoteGraphBackendLoader:
    """Utility class to load triplets into a RAG Backend."""
    path: str
    datatype: RemoteDataType
    graph_store_type: Type[ConvertableGraphStore]
    feature_store_type: Type[ConvertableFeatureStore]

    def load(self, pid: Optional[int] = None) -> RemoteGraphBackend:
        if self.datatype == RemoteDataType.DATA:
            data_obj = torch.load(self.path)
            graph_store = self.graph_store_type.from_data(
                edge_id=data_obj['edge_id'], edge_index=data_obj.edge_index,
                num_nodes=data_obj.num_nodes)
            feature_store = self.feature_store_type.from_data(
                node_id=data_obj['node_id'], x=data_obj.x,
                edge_id=data_obj['edge_id'], edge_attr=data_obj.edge_attr)
        elif self.datatype == RemoteDataType.PARTITION:
            if pid is None:
                assert pid is not None, \
                    "Partition ID must be defined for loading from a " \
                    + "partitioned store."
            graph_store = self.graph_store_type.from_partition(self.path, pid)
            feature_store = self.feature_store_type.from_partition(
                self.path, pid)
        else:
            raise NotImplementedError
        return (feature_store, graph_store)


# TODO: make profilable
def create_remote_backend_from_triplets(
    triplets: Iterable[TripletLike], node_embedding_model: Module,
    edge_embedding_model: Module | None = None,
    graph_db: Type[ConvertableGraphStore] = LocalGraphStore,
    feature_db: Type[ConvertableFeatureStore] = LocalFeatureStore,
    node_method_to_call: str = "forward",
    edge_method_to_call: str | None = None,
    pre_transform: Callable[[TripletLike], TripletLike] | None = None,
    path: str = '', n_parts: int = 1,
    node_method_kwargs: Optional[Dict[str, Any]] = None,
    edge_method_kwargs: Optional[Dict[str, Any]] = None
) -> RemoteGraphBackendLoader:
    """Utility function that can be used to create a RAG Backend from triplets.

    Args:
        triplets (Iterable[TripletLike]): Triplets to load into the RAG
            Backend.
        node_embedding_model (Module): Model to embed nodes into a feature
            space.
        edge_embedding_model (Module | None, optional): Model to embed edges
            into a feature space. Defaults to the node model.
        graph_db (Type[ConvertableGraphStore], optional): GraphStore class to
            use. Defaults to LocalGraphStore.
        feature_db (Type[ConvertableFeatureStore], optional): FeatureStore
            class to use. Defaults to LocalFeatureStore.
        node_method_to_call (str, optional): method to call for embeddings on
            the node model. Defaults to "forward".
        edge_method_to_call (str | None, optional): method to call for
            embeddings on the edge model. Defaults to the node method.
        pre_transform (Callable[[TripletLike], TripletLike] | None, optional):
            optional preprocessing function for triplets. Defaults to None.
        path (str, optional): path to save resulting stores. Defaults to ''.
        n_parts (int, optional): Number of partitions to store in.
            Defaults to 1.
        node_method_kwargs (Optional[Dict[str, Any]], optional): args to pass
            into node encoding method. Defaults to None.
        edge_method_kwargs (Optional[Dict[str, Any]], optional): args to pass
            into edge encoding method. Defaults to None.

    Returns:
        RemoteGraphBackendLoader: Loader to load RAG backend from disk or
            memory.
    """
    # Will return attribute errors for missing attributes
    if not issubclass(graph_db, ConvertableGraphStore):
        graph_db.from_data  # noqa: B018
        graph_db.from_hetero_data  # noqa: B018
        graph_db.from_partition  # noqa: B018
    elif not issubclass(feature_db, ConvertableFeatureStore):
        feature_db.from_data  # noqa: B018
        feature_db.from_hetero_data  # noqa: B018
        feature_db.from_partition  # noqa: B018

    # Resolve callable methods
    node_method_kwargs = node_method_kwargs \
        if node_method_kwargs is not None else dict()

    edge_embedding_model = edge_embedding_model \
        if edge_embedding_model is not None else node_embedding_model
    edge_method_to_call = edge_method_to_call \
        if edge_method_to_call is not None else node_method_to_call
    edge_method_kwargs = edge_method_kwargs \
        if edge_method_kwargs is not None else node_method_kwargs

    # These will return AttributeErrors if they don't exist
    node_model = getattr(node_embedding_model, node_method_to_call)
    edge_model = getattr(edge_embedding_model, edge_method_to_call)

    indexer = LargeGraphIndexer.from_triplets(triplets,
                                              pre_transform=pre_transform)

    node_feats = node_model(indexer.get_node_features(), **node_method_kwargs)
    indexer.add_node_feature('x', node_feats)

    edge_feats = edge_model(
        indexer.get_unique_edge_features(feature_name=EDGE_RELATION),
        **edge_method_kwargs)
    indexer.add_edge_feature(new_feature_name="edge_attr",
                             new_feature_vals=edge_feats,
                             map_from_feature=EDGE_RELATION)

    data = indexer.to_data(node_feature_name='x',
                           edge_feature_name='edge_attr')

    if n_parts == 1:
        torch.save(data, path)
        return RemoteGraphBackendLoader(path, RemoteDataType.DATA, graph_db,
                                        feature_db)
    else:
        partitioner = Partitioner(data=data, num_parts=n_parts, root=path)
        partitioner.generate_partition()
        return RemoteGraphBackendLoader(path, RemoteDataType.PARTITION,
                                        graph_db, feature_db)
