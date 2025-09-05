import os
import tempfile
from typing import List

import torch

from torch_geometric.data import Data
from torch_geometric.llm.utils.backend_utils import (
    create_graph_from_triples,
    create_remote_backend_from_graph_data,
)


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
