import os
from dataclasses import dataclass
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Protocol,
    Tuple,
    Type,
    Union,
    no_type_check,
    runtime_checkable,
)

import numpy as np
import torch
from torch import Tensor
from torch.nn import Module

from torch_geometric.data import Data, FeatureStore, GraphStore
from torch_geometric.distributed import (
    LocalFeatureStore,
    LocalGraphStore,
    Partitioner,
)
from torch_geometric.llm.large_graph_indexer import (
    EDGE_RELATION,
    LargeGraphIndexer,
    TripletLike,
)
from torch_geometric.llm.models import SentenceTransformer
from torch_geometric.typing import EdgeType, NodeType

try:
    from pandas import DataFrame
except ImportError:
    DataFrame = None
RemoteGraphBackend = Tuple[FeatureStore, GraphStore]

# TODO: Make everything compatible with Hetero graphs aswell


def preprocess_triplet(triplet: TripletLike) -> TripletLike:
    h, r, t = triplet
    return str(h).lower(), str(r).lower(), str(t).lower()


@no_type_check
def retrieval_via_pcst(
    data: Data,
    q_emb: Tensor,
    textual_nodes: Any,
    textual_edges: Any,
    topk: int = 3,
    topk_e: int = 5,
    cost_e: float = 0.5,
    num_clusters: int = 1,
) -> Tuple[Data, str]:

    # skip PCST for bad graphs
    booly = data.edge_attr is None or data.edge_attr.numel() == 0
    booly = booly or data.x is None or data.x.numel() == 0
    booly = booly or data.edge_index is None or data.edge_index.numel() == 0
    if not booly:
        c = 0.01

        from pcst_fast import pcst_fast

        root = -1
        pruning = 'gw'
        verbosity_level = 0
        if topk > 0:
            n_prizes = torch.nn.CosineSimilarity(dim=-1)(q_emb, data.x)
            topk = min(topk, data.num_nodes)
            _, topk_n_indices = torch.topk(n_prizes, topk, largest=True)

            n_prizes = torch.zeros_like(n_prizes)
            n_prizes[topk_n_indices] = torch.arange(topk, 0, -1).float()
        else:
            n_prizes = torch.zeros(data.num_nodes)

        if topk_e > 0:
            e_prizes = torch.nn.CosineSimilarity(dim=-1)(q_emb, data.edge_attr)
            topk_e = min(topk_e, e_prizes.unique().size(0))

            topk_e_values, _ = torch.topk(e_prizes.unique(), topk_e,
                                          largest=True)
            e_prizes[e_prizes < topk_e_values[-1]] = 0.0
            last_topk_e_value = topk_e
            for k in range(topk_e):
                indices = e_prizes == topk_e_values[k]
                value = min((topk_e - k) / sum(indices), last_topk_e_value - c)
                e_prizes[indices] = value
                last_topk_e_value = value * (1 - c)
            # reduce the cost of the edges so that at least one edge is chosen
            cost_e = min(cost_e, e_prizes.max().item() * (1 - c / 2))
        else:
            e_prizes = torch.zeros(data.num_edges)

        costs = []
        edges = []
        virtual_n_prizes = []
        virtual_edges = []
        virtual_costs = []
        mapping_n = {}
        mapping_e = {}
        for i, (src, dst) in enumerate(data.edge_index.t().numpy()):
            prize_e = e_prizes[i]
            if prize_e <= cost_e:
                mapping_e[len(edges)] = i
                edges.append((src, dst))
                costs.append(cost_e - prize_e)
            else:
                virtual_node_id = data.num_nodes + len(virtual_n_prizes)
                mapping_n[virtual_node_id] = i
                virtual_edges.append((src, virtual_node_id))
                virtual_edges.append((virtual_node_id, dst))
                virtual_costs.append(0)
                virtual_costs.append(0)
                virtual_n_prizes.append(prize_e - cost_e)

        prizes = np.concatenate([n_prizes, np.array(virtual_n_prizes)])
        num_edges = len(edges)
        if len(virtual_costs) > 0:
            costs = np.array(costs + virtual_costs)
            edges = np.array(edges + virtual_edges)

        vertices, edges = pcst_fast(edges, prizes, costs, root, num_clusters,
                                    pruning, verbosity_level)

        selected_nodes = vertices[vertices < data.num_nodes]
        selected_edges = [mapping_e[e] for e in edges if e < num_edges]
        virtual_vertices = vertices[vertices >= data.num_nodes]
        if len(virtual_vertices) > 0:
            virtual_vertices = vertices[vertices >= data.num_nodes]
            virtual_edges = [mapping_n[i] for i in virtual_vertices]
            selected_edges = np.array(selected_edges + virtual_edges)

        edge_index = data.edge_index[:, selected_edges]
        selected_nodes = np.unique(
            np.concatenate(
                [selected_nodes, edge_index[0].numpy(),
                 edge_index[1].numpy()]))

        n = textual_nodes.iloc[selected_nodes]
        e = textual_edges.iloc[selected_edges]
    else:
        n = textual_nodes
        e = textual_edges

    desc = n.to_csv(index=False) + '\n' + e.to_csv(
        index=False, columns=['src', 'edge_attr', 'dst'])

    if booly:
        return data, desc

    mapping = {n: i for i, n in enumerate(selected_nodes.tolist())}
    src = [mapping[i] for i in edge_index[0].tolist()]
    dst = [mapping[i] for i in edge_index[1].tolist()]

    # HACK Added so that the subset of nodes and edges selected can be tracked
    node_idx = np.array(data.node_idx)[selected_nodes]
    edge_idx = np.array(data.edge_idx)[selected_edges]

    data = Data(
        x=data.x[selected_nodes],
        edge_index=torch.tensor([src, dst]).to(torch.long),
        edge_attr=data.edge_attr[selected_edges],
        # HACK: track subset of selected nodes/edges
        node_idx=node_idx,
        edge_idx=edge_idx,
    )

    return data, desc


def batch_knn(query_enc: Tensor, embeds: Tensor,
              k: int) -> Iterator[Tuple[Tensor, Tensor]]:
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

    def __del__(self) -> None:
        if os.path.exists(self.path):
            os.remove(self.path)


def create_graph_from_triples(
    triples: Iterable[TripletLike],
    embedding_model: Union[Module, Callable],
    embedding_method_kwargs: Optional[Dict[str, Any]] = None,
    pre_transform: Optional[Callable[[TripletLike], TripletLike]] = None,
) -> Data:
    """Utility function that can be used to create a graph from triples."""
    # Resolve callable methods
    embedding_method_kwargs = embedding_method_kwargs \
        if embedding_method_kwargs is not None else dict()

    indexer = LargeGraphIndexer.from_triplets(triples,
                                              pre_transform=pre_transform)
    node_feats = embedding_model(indexer.get_unique_node_features(),
                                 **embedding_method_kwargs)
    indexer.add_node_feature('x', node_feats)

    edge_feats = embedding_model(
        indexer.get_unique_edge_features(feature_name=EDGE_RELATION),
        **embedding_method_kwargs)
    indexer.add_edge_feature(new_feature_name="edge_attr",
                             new_feature_vals=edge_feats,
                             map_from_feature=EDGE_RELATION)

    data = indexer.to_data(node_feature_name='x',
                           edge_feature_name='edge_attr')
    data = data.to("cpu")
    return data


def create_remote_backend_from_graph_data(
    graph_data: Data,
    graph_db: Type[ConvertableGraphStore] = LocalGraphStore,
    feature_db: Type[ConvertableFeatureStore] = LocalFeatureStore,
    path: str = '',
    n_parts: int = 1,
) -> RemoteGraphBackendLoader:
    """Utility function that can be used to create a RAG Backend from triples.

    Args:
        graph_data (Data): Graph data to load into the RAG Backend.
        graph_db (Type[ConvertableGraphStore], optional): GraphStore class to
            use. Defaults to LocalGraphStore.
        feature_db (Type[ConvertableFeatureStore], optional): FeatureStore
            class to use. Defaults to LocalFeatureStore.
        path (str, optional): path to save resulting stores. Defaults to ''.
        n_parts (int, optional): Number of partitons to store in.
            Defaults to 1.

    Returns:
        RemoteGraphBackendLoader: Loader to load RAG backend from disk or
            memory.
    """
    # Will return attribute errors for missing attributes
    if not issubclass(graph_db, ConvertableGraphStore):
        _ = graph_db.from_data
        _ = graph_db.from_hetero_data
        _ = graph_db.from_partition
    elif not issubclass(feature_db, ConvertableFeatureStore):
        _ = feature_db.from_data
        _ = feature_db.from_hetero_data
        _ = feature_db.from_partition

    if n_parts == 1:
        torch.save(graph_data, path)
        return RemoteGraphBackendLoader(path, RemoteDataType.DATA, graph_db,
                                        feature_db)
    else:
        partitioner = Partitioner(data=graph_data, num_parts=n_parts,
                                  root=path)
        partitioner.generate_partition()
        return RemoteGraphBackendLoader(path, RemoteDataType.PARTITION,
                                        graph_db, feature_db)


def make_pcst_filter(triples: List[Tuple[str, str,
                                         str]], model: SentenceTransformer,
                     topk: int = 5, topk_e: int = 5, cost_e: float = 0.5,
                     num_clusters: int = 1) -> Callable[[Data, str], Data]:
    """Creates a PCST (Prize Collecting Tree) filter.

    :param triples: List of triples (head, relation, tail) representing KG data
    :param model: SentenceTransformer model for embedding text
    :param topk: Number of top-K results to return (default: 5)
    :param topk_e: Number of top-K entity results to return (default: 5)
    :param cost_e: Cost of edges (default: 0.5)
    :param num_clusters: Number of connected components in the PCST output.
    :return: PCST Filter function
    """
    if DataFrame is None:
        raise Exception("PCST requires `pip install pandas`"
                        )  # Check if pandas is installed

    # Remove duplicate triples to ensure unique set
    triples = list(dict.fromkeys(triples))

    # Initialize empty list to store nodes (entities) from triples
    nodes = []

    # Iterate over triples to extract unique nodes (entities)
    for h, _, t in triples:
        for node in (h, t):  # Extract head and tail entities from each triple
            nodes.append(node)

    # Remove duplicates and create final list of unique nodes
    nodes = list(dict.fromkeys(nodes))

    # Create full list of textual nodes (entities) for filtering
    full_textual_nodes = nodes

    def apply_retrieval_via_pcst(
            graph: Data,  # Input graph data
            query: str,  # Search query
    ) -> Data:
        """Applies PCST filtering for retrieval.

        :param graph: Input graph data
        :param query: Search query
        :return: Retrieved graph/query data
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
                                             topk=topk, topk_e=topk_e,
                                             cost_e=cost_e,
                                             num_clusters=num_clusters)
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
