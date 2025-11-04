import os
import tempfile
from typing import List, Tuple

import torch

from torch_geometric.data import Data
from torch_geometric.llm.utils.backend_utils import (
    batch_knn,
    create_graph_from_triples,
    create_remote_backend_from_graph_data,
    make_pcst_filter,
    preprocess_triplet,
    retrieval_via_pcst,
)
from torch_geometric.testing import withPackage


def test_preprocess_triplet():
    triplet = ('Alice', 'works with', 'Bob')
    processed = preprocess_triplet(triplet)
    assert processed == ('alice', 'works with', 'bob')


def test_batch_knn():
    query_embeddings = torch.randn(2, 64)
    candidate_embeddings = torch.randn(10, 64)
    k = 3
    top_k_indices, top_k_scores = batch_knn(
        query_embeddings,
        candidate_embeddings,
        k,
    )
    assert top_k_indices[0].size() == (k, )
    assert top_k_indices[1].size() == (1, 64)
    assert top_k_scores[0].size() == (k, )
    assert top_k_scores[1].size() == (1, 64)


"""Test retrieval_via_pcst"""


@withPackage('pandas')
def create_mock_data(num_nodes=3, num_edges=2):
    import pandas as pd
    x = torch.randn(num_nodes, 16)
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    edge_attr = torch.randn(num_edges, 16)
    node_idx = list(range(num_nodes))
    edge_idx = list(range(num_edges))

    textual_nodes = pd.DataFrame({
        'node_id': node_idx,
        'text': [f"Node {i}" for i in node_idx]
    })
    textual_edges = pd.DataFrame({
        'src': edge_index[0].tolist(),
        'dst': edge_index[1].tolist(),
        'edge_attr': [f"Edge {i}" for i in edge_idx]
    })

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                node_idx=node_idx,
                edge_idx=edge_idx), textual_nodes, textual_edges


@withPackage('pandas')
def test_empty_graph():
    import pandas as pd

    # w/o node and edge
    bad_data = Data(x=None, edge_index=None, edge_attr=None)
    textual_nodes = pd.DataFrame({'node_id': [], 'text': []})
    textual_edges = pd.DataFrame({'src': [], 'dst': [], 'edge_attr': []})
    q_emb = torch.randn(1, 16)

    result_data, desc = retrieval_via_pcst(bad_data, q_emb, textual_nodes,
                                           textual_edges)

    assert result_data == bad_data
    assert desc.strip() == 'node_id,text\n\nsrc,edge_attr,dst'


def test_topk_zero():
    data, textual_nodes, textual_edges = create_mock_data()
    q_emb = torch.randn(1, 16)

    result_data, desc = retrieval_via_pcst(data, q_emb, textual_nodes,
                                           textual_edges, topk=0, topk_e=0)

    assert isinstance(result_data, Data)


"""Test make_pcst_filter"""


class MockSentenceTransformer:
    def encode(self, sentences, **kwargs):
        return torch.randn(len(sentences), 32)


def create_mock_graph_and_triples():
    triples: List[Tuple[str, str, str]] = [("Alice", "works_at", "Google"),
                                           ("Bob", "works_at", "Meta"),
                                           ("Alice", "knows", "Bob")]

    # Alice=0, Bob=1, Google=2, Meta=3
    node_idx = [0, 1, 2, 3]
    edge_idx = [0, 1, 2]

    x = torch.randn(4, 32)
    edge_index = torch.tensor([[0, 1, 0], [2, 3, 1]], dtype=torch.long)
    edge_attr = torch.randn(3, 32)

    graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                 node_idx=node_idx, edge_idx=edge_idx)

    return triples, graph


def test_apply_retrieval_via_pcst_isolated_node():
    triples, graph = create_mock_graph_and_triples()
    model = MockSentenceTransformer()

    mock_out_graph = Data(x=graph.x[:1],
                          edge_index=torch.empty(2, 0, dtype=torch.long),
                          edge_attr=torch.empty(0, 32))
    mock_out_graph.node_idx = [0]
    mock_out_graph.edge_idx = []

    filter_fn = make_pcst_filter(triples, model)
    result = filter_fn(graph, "Who is Alice?")

    assert result.triples == [
        ('Alice', 'works_at', 'Google'),
        ('Bob', 'works_at', 'Meta'),
        ('Alice', 'knows', 'Bob'),
    ]


class MockEmbeddingModel:
    """Mock embedding model for testing."""
    def __init__(self, embed_dim: int = 64):
        self.embed_dim = embed_dim

    def __call__(self, texts: List[str], **kwargs) -> torch.Tensor:
        """Mock embedding generation - creates deterministic embeddings."""
        # Create simple hash-based embeddings for reproducible testing
        if len(texts) == 0:
            return torch.empty(0, self.embed_dim)
        embeddings = []
        for text in texts:
            # Simple deterministic embedding based on text hash
            hash_val = hash(text)
            # Use the hash to create a reproducible embedding
            torch.manual_seed(abs(hash_val) % 2**31)
            embedding = torch.randn(self.embed_dim)
            embeddings.append(embedding)
        return torch.stack(embeddings)


class TestCreateGraphFromTriples:
    """Test suite for create_graph_from_triples function."""
    def setup_method(self):
        """Set up test fixtures."""
        self.sample_triples = [('Alice', 'works with', 'Bob'),
                               ('Alice', 'leads', 'Carol'),
                               ('Carol', 'works with', 'Dave')]
        self.mock_embedding_model = MockEmbeddingModel(embed_dim=32)

    def test_create_graph_basic_functionality(self):
        """Test basic functionality of create_graph_from_triples."""
        result = create_graph_from_triples(
            triples=self.sample_triples,
            embedding_model=self.mock_embedding_model)

        # Verify result is a Data object
        assert isinstance(result, Data)

        x = result.x
        edge_attr = result.edge_attr
        assert x.shape == (4, 32)
        assert edge_attr.shape == (3, 32)
        for t in self.sample_triples:
            assert self.mock_embedding_model([t[0]]) in x
            assert self.mock_embedding_model([t[2]]) in x
            assert self.mock_embedding_model([t[1]]) in edge_attr

        expected_edge_index = torch.tensor([[0, 0, 2], [1, 2, 3]])
        assert torch.allclose(result.edge_index, expected_edge_index)

    def test_create_graph_empty_triples(self):
        """Test create_graph_from_triples with empty triples list."""
        empty_triples = []

        result = create_graph_from_triples(
            triples=empty_triples, embedding_model=self.mock_embedding_model)

        # Should create an empty graph
        assert isinstance(result, Data)
        assert result.num_nodes == 0
        assert result.num_edges == 0


class TestCreateRemoteBackendFromGraphData:
    """Test suite for create_remote_backend_from_graph_data function."""
    def setup_method(self):
        """Set up test fixtures."""
        self.sample_triples = [('Alice', 'works with', 'Bob'),
                               ('Alice', 'leads', 'Carol'),
                               ('Carol', 'works with', 'Dave')]
        self.mock_embedding_model = MockEmbeddingModel(embed_dim=32)

        # Create sample graph data using create_graph_from_triples
        self.sample_graph_data = create_graph_from_triples(
            triples=self.sample_triples,
            embedding_model=self.mock_embedding_model)

    def test_create_backend_data_load(self):
        """Test that data integrity is preserved in backend creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = os.path.join(temp_dir, "test_graph.pt")

            loader = create_remote_backend_from_graph_data(
                graph_data=self.sample_graph_data, path=save_path, n_parts=1)

            # Load and verify data
            feature_store, graph_store = loader.load()

            # Check that the original graph structure is preserved
            loaded_data = torch.load(save_path, weights_only=False)

            # Verify basic properties match
            assert loaded_data.num_nodes == self.sample_graph_data.num_nodes
            assert loaded_data.num_edges == self.sample_graph_data.num_edges

            # Verify tensors match
            assert torch.allclose(loaded_data.x, self.sample_graph_data.x)
            assert torch.allclose(loaded_data.edge_index,
                                  self.sample_graph_data.edge_index)
            assert torch.allclose(loaded_data.edge_attr,
                                  self.sample_graph_data.edge_attr)
