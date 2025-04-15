from dataclasses import dataclass
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
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
    Data,
    FeatureStore,
    GraphStore,
    LargeGraphIndexer,
    TripletLike,
)
from torch_geometric.data.large_graph_indexer import EDGE_RELATION
from torch_geometric.datasets.web_qsp_dataset import retrieval_via_pcst
from torch_geometric.distributed import (
    LocalFeatureStore,
    LocalGraphStore,
    Partitioner,
)
from torch_geometric.nn.nlp import SentenceTransformer
from torch_geometric.typing import EdgeType, NodeType

try:
    from pandas import DataFrame
except ImportError:
    DataFrame = None
RemoteGraphBackend = Tuple[FeatureStore, GraphStore]

# TODO: Make everything compatible with Hetero graphs aswell


# (TODO) once Zacks webqsp PR is merged
# https://github.com/pyg-team/pytorch_geometric/pull/9806
# update WebQSP in this branch to use preprocess_triplet from here
def preprocess_triplet(triplet: TripletLike) -> TripletLike:
    h, r, t = triplet
    return str(h).lower(), str(r).lower(), str(t).lower()

def batch_knn(query_enc: Tensor, embeds: Tensor,
              k: int) -> Iterator[InputNodes]:
    from torchmetrics.functional import pairwise_cosine_similarity
    prizes = pairwise_cosine_similarity(query_enc, embeds.to(query_enc.device))
    topk = min(k, len(embeds))
    for i, q in enumerate(prizes):
        _, indices = torch.topk(q, topk, largest=True)
        yield indices, query_enc[i].unsqueeze(0)

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
            data_obj = torch.load(self.path, weights_only=False)
            # is_sorted=true since assume nodes come sorted from indexer
            graph_store = self.graph_store_type.from_data(
                edge_id=data_obj['edge_id'], edge_index=data_obj.edge_index,
                num_nodes=data_obj.num_nodes, is_sorted=True)
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
    triplets: Iterable[TripletLike],
    node_embedding_model: Module,
    edge_embedding_model: Module | None = None,
    graph_db: Type[ConvertableGraphStore] = LocalGraphStore,
    feature_db: Type[ConvertableFeatureStore] = LocalFeatureStore,
    node_method_to_call: str = "forward",
    edge_method_to_call: str | None = None,
    pre_transform: Callable[[TripletLike], TripletLike] | None = None,
    path: str = '',
    n_parts: int = 1,
    node_method_kwargs: Optional[Dict[str, Any]] = None,
    edge_method_kwargs: Optional[Dict[str, Any]] = None,
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
        n_parts (int, optional): Number of partitons to store in.
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
        getattr(graph_db, "from_data")
        getattr(graph_db, "from_hetero_data")
        getattr(graph_db, "from_partition")
    elif not issubclass(feature_db, ConvertableFeatureStore):
        getattr(feature_db, "from_data")
        getattr(feature_db, "from_hetero_data")
        getattr(feature_db, "from_partition")

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
    node_feats = node_model(indexer.get_unique_node_features(),
                            **node_method_kwargs)
    indexer.add_node_feature('x', node_feats)

    edge_feats = edge_model(
        indexer.get_unique_edge_features(feature_name=EDGE_RELATION),
        **edge_method_kwargs)
    indexer.add_edge_feature(new_feature_name="edge_attr",
                             new_feature_vals=edge_feats,
                             map_from_feature=EDGE_RELATION)

    data = indexer.to_data(node_feature_name='x',
                           edge_feature_name='edge_attr')
    data = data.to("cpu")
    if n_parts == 1:
        torch.save(data, path)
        return RemoteGraphBackendLoader(path, RemoteDataType.DATA, graph_db,
                                        feature_db)
    else:
        partitioner = Partitioner(data=data, num_parts=n_parts, root=path)
        partitioner.generate_partition()
        return RemoteGraphBackendLoader(path, RemoteDataType.PARTITION,
                                        graph_db, feature_db)


def make_pcst_filter(triples: List[Tuple[str, str, str]],
                     model: SentenceTransformer) -> None:
    """Creates a PCST (Prize Collecting Tree) filter.

    :param triples: List of triples (head, relation, tail) representing knowledge graph data
    :param model: SentenceTransformer model for generating semantic representations
    :return: None
    """
    if DataFrame is None:
        raise Exception("PCST requires `pip install pandas`"
                        )  # Check if pandas is installed

    # Remove duplicate triples to ensure unique set
    triples = list(dict.fromkeys(triples))

    # Initialize empty list to store nodes (entities) from triples
    nodes = []

    # Iterate over triples to extract unique nodes (entities)
    for h, r, t in triples:
        for node in (h, t):  # Extract head and tail entities from each triple
            # Add node to list if not already present
            nodes.append(node)

    # Remove duplicates and create final list of unique nodes
    nodes = list(dict.fromkeys(nodes))

    # Create full list of textual nodes (entities) for filtering
    full_textual_nodes = nodes

    def apply_retrieval_via_pcst(
        graph: Data,  # Input graph data
        query: str,  # Search query
        topk: int = 5,  # Number of top-K results to return (default: 5)
        topk_e: int = 5,
        cost_e: float = 0.5,
        num_clusters: int = 1,
    ) -> Tuple[Data, str]:
        """Applies PCST filtering for retrieval.
        PCST = Prize Collecting Steiner Tree

        :param graph: Input graph data
        :param query: Search query
        :param topk: Number of top-K results to return (default: 5)
        :param topk_e: Number of top-K entity results to return (default: 5)
        :param cost_e: Cost of edges (default: 0.5)
        :param num_clusters: the number of connected components in the PCST output.
        :return: Retrieved graph data and query result
        """
        # PCST relies on numpy and pcst_fast pypi libs, hence to("cpu")
        q_emb = model.encode([query]).to("cpu")
        textual_nodes = [(int(i), full_textual_nodes[i])
                         for i in graph["node_idx"]]
        textual_nodes = DataFrame(textual_nodes,
                                  columns=["node_id", "node_attr"])
        textual_edges = [triples[i] for i in graph["edge_idx"]]
        textual_edges = DataFrame(textual_edges,
                                  columns=["src", "edge_attr", "dst"])
        out_graph, desc = retrieval_via_pcst(graph.to(q_emb.device), q_emb,
                                             textual_nodes, textual_edges,
                                             topk, topk_e, cost_e,
                                             num_clusters)
        out_graph["desc"] = desc
        where_trips_start = desc.find("src,edge_attr,dst")
        parsed_trips = []
        for trip in desc[where_trips_start + 18:-1].split("\n"):
            parsed_trips.append(tuple(trip.split(",")))

        # Handle case where PCST returns an isolated node
        """
        TODO find a better solution since these failed subgraphs
        severely hurt accuracy.
        """
        if str(parsed_trips) == "[('',)]" or out_graph.edge_index.numel() == 0:
            out_graph["triples"] = []
        else:
            out_graph["triples"] = parsed_trips
        out_graph["question"] = query
        return out_graph

    return apply_retrieval_via_pcst
