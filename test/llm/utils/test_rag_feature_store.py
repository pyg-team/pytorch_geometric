from unittest.mock import Mock, patch

import pytest
import torch

from torch_geometric.data import Data
from torch_geometric.llm.utils.feature_store import KNNRAGFeatureStore
from torch_geometric.sampler import SamplerOutput
from torch_geometric.testing.decorators import onlyRAG


class TestKNNRAGFeatureStore:
    """Test suite for KNNRAGFeatureStore methods."""
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_encoder = Mock()
        self.mock_encoder.encode = Mock()
        self.mock_encoder.to = Mock(return_value=self.mock_encoder)
        self.mock_encoder.eval = Mock()

        self.config = {"k_nodes": 5, "encoder_model": self.mock_encoder}
        self.sample_x = torch.randn(40, 128)  # 40 nodes, 128 features
        self.sample_edge_attr = torch.randn(40, 64)  # 40 edges, 64 features

    def test_bad_config(self):
        """Test bad config initialization."""
        with pytest.raises(ValueError, match="Required config parameter"):
            store = KNNRAGFeatureStore()
            store.config = {}

    def create_feature_store(self):
        """Create a FeatureStore with mocked dependencies."""
        store = KNNRAGFeatureStore()

        store.config = self.config

        # Mock the tensor storage
        store.put_tensor(self.sample_x, group_name=None, attr_name='x')
        store.put_tensor(self.sample_edge_attr, group_name=(None, None),
                         attr_name='edge_attr')

        return store

    @onlyRAG
    def test_retrieve_seed_nodes_single_query(self):
        """Test retrieve_seed_nodes with a single query."""
        store = self.create_feature_store()

        # Mock the encoder output and batch_knn
        query_text = "test query"
        mock_query_enc = torch.randn(1, 128)
        self.mock_encoder.encode.return_value = mock_query_enc

        expected_indices = torch.tensor([0, 3, 7, 2, 9])

        with patch('torch_geometric.utils.rag.feature_store.batch_knn'
                   ) as mock_batch_knn:
            # Mock batch_knn to return an iterator
            def mock_generator():
                yield (expected_indices, mock_query_enc)

            mock_batch_knn.return_value = mock_generator()

            result, query_enc = store.retrieve_seed_nodes(query_text)

            # Verify encoder was called correctly
            self.mock_encoder.encode.assert_called_once_with([query_text])

            # Verify batch_knn was called correctly
            mock_batch_knn.assert_called_once()
            args = mock_batch_knn.call_args[0]
            assert torch.equal(args[0], mock_query_enc)
            assert torch.equal(args[1], self.sample_x)
            assert args[2] == 5  # k_nodes

            # Verify results
            assert torch.equal(result, expected_indices)
            assert torch.equal(query_enc, mock_query_enc)

    @onlyRAG
    def test_retrieve_seed_nodes_multiple_queries(self):
        """Test retrieve_seed_nodes with multiple queries."""
        store = self.create_feature_store()

        queries = ["query 1", "query 2"]
        mock_query_enc = torch.randn(2, 128)
        self.mock_encoder.encode.return_value = mock_query_enc

        expected_indices = [
            torch.tensor([1, 4, 6, 8, 0]),
            torch.tensor([0, 3, 7, 2, 9])
        ]

        with patch('torch_geometric.utils.rag.feature_store.batch_knn'
                   ) as mock_batch_knn:

            def mock_generator():
                for i in range(len(expected_indices)):
                    yield (expected_indices[i], mock_query_enc[i])

            mock_batch_knn.return_value = mock_generator()

            out_dict = store.retrieve_seed_nodes(queries)

            # Verify encoder was called with the list directly
            self.mock_encoder.encode.assert_called_once_with(queries)

            # Verify results
            for i, query in enumerate(queries):
                result, query_enc = out_dict[query]
                assert torch.equal(result, expected_indices[i])
                assert torch.equal(query_enc, mock_query_enc[i])

    @pytest.mark.parametrize("induced", [True, False])
    def test_load_subgraph_valid_sample(self, induced):
        """Test load_subgraph with valid SamplerOutput."""
        store = self.create_feature_store()

        # Create a mock SamplerOutput
        sample = SamplerOutput(node=torch.tensor([6, 7, 8, 9]),
                               row=torch.tensor([0, 1, 2]),
                               col=torch.tensor([1, 2, 3]),
                               edge=torch.tensor([0, 1, 2]), batch=None)

        expected_edge_indices = torch.tensor([[0, 1, 2], [1, 2, 3]]) \
            if induced else torch.tensor([[6, 7, 8], [7, 8, 9]])

        result = store.load_subgraph(sample, induced=induced)

        # Verify result is a Data object
        assert isinstance(result, Data)

        # Verify edge attributes are correctly extracted
        expected_edge_attr = self.sample_edge_attr[torch.tensor([0, 1, 2])]
        assert torch.equal(result.edge_attr, expected_edge_attr)
        assert torch.equal(result.edge_index, expected_edge_indices)
        if induced:
            assert torch.equal(result.node_idx, sample.node)
            assert torch.equal(result.edge_idx, sample.edge)
