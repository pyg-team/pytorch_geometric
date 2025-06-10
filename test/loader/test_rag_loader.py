import torch
import pytest
from unittest.mock import Mock
from typing import Any, Dict

from torch_geometric.data import Data
from torch_geometric.loader.rag_loader import RAGQueryLoader
from torch_geometric.sampler import SamplerOutput
from torch_geometric.utils.rag.vectorrag import VectorRetriever


class MockRAGFeatureStore:
    """Mock implementation of RAGFeatureStore protocol for testing."""
    
    def __init__(self):
        self._config = {}
        self.x = torch.randn(10, 64)  # Sample node features
    
    def retrieve_seed_nodes(self, query: Any, **kwargs):
        """Mock retrieve_seed_nodes method."""
        seed_nodes = torch.tensor([0, 1, 2, 3, 4])
        query_enc = torch.randn(1, 64)
        return seed_nodes, query_enc
    
    @property
    def config(self) -> Dict[str, Any]:
        return self._config
    
    @config.setter
    def config(self, config: Dict[str, Any]):
        if config is None:
            raise ValueError("Config cannot be None")
        if 'a' not in config:
            raise ValueError("Required config parameter 'a' not found")
        self._config = config
    
    def retrieve_seed_edges(self, query: Any, **kwargs):
        """Mock retrieve_seed_edges method."""
        return torch.tensor([[0, 1], [1, 2], [2, 3]])
    
    def load_subgraph(self, sample):
        """Mock load_subgraph method."""
        data = Data()
        data.edge_idx = torch.tensor([0, 1, 2])
        return data


class MockRAGGraphStore:
    """Mock implementation of RAGGraphStore protocol for testing."""
    
    def __init__(self):
        self._config = {}
        self.edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]])
    
    def sample_subgraph(self, seed_nodes, seed_edges=None, **kwargs):
        """Mock sample_subgraph method."""
        return SamplerOutput(
            node=seed_nodes,
            row=torch.tensor([0, 1, 2]),
            col=torch.tensor([1, 2, 3]),
            edge=torch.tensor([0, 1, 2]),
            batch=None
        )
    
    @property
    def config(self) -> Dict[str, Any]:
        return self._config
    
    @config.setter
    def config(self, config: Dict[str, Any]):
        if config is None:
            raise ValueError("Config cannot be None")
        if 'b' not in config:
            raise ValueError("Required config parameter 'b' not found")
        self._config = config
    
    def register_feature_store(self, feature_store):
        """Mock register_feature_store method."""
        pass


class TestRAGQueryLoader:
    """Test suite for RAGQueryLoader."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.mock_feature_store = MockRAGFeatureStore()
        self.mock_graph_store = MockRAGGraphStore()
        self.graph_data = (self.mock_feature_store, self.mock_graph_store)
        
        # Sample config
        self.sample_config = {
            "a": 5,
            "b": [10, 5],
            "c": "test_value"
        }
    
    def test_initialization_basic(self):
        """Test basic initialization of RAGQueryLoader."""
        loader = RAGQueryLoader(self.graph_data, config=self.sample_config)
        
        assert loader.feature_store == self.mock_feature_store
        assert loader.graph_store == self.mock_graph_store
        assert loader.vector_retriever is None
        assert loader.augment_query is False
        assert loader.subgraph_filter is None
        assert loader.config == self.sample_config
    
    def test_initialization_with_all_params(self):
        """Test initialization with all parameters."""
        mock_vector_retriever = Mock(spec=VectorRetriever)
        mock_subgraph_filter = Mock()
        
        loader = RAGQueryLoader(
            graph_data=self.graph_data,
            subgraph_filter=mock_subgraph_filter,
            augment_query=True,
            vector_retriever=mock_vector_retriever,
            config=self.sample_config
        )
        
        assert loader.feature_store == self.mock_feature_store
        assert loader.graph_store == self.mock_graph_store
        assert loader.vector_retriever == mock_vector_retriever
        assert loader.augment_query is True
        assert loader.subgraph_filter == mock_subgraph_filter
        assert loader.config == self.sample_config
    

    def test_bad_config(self):
        """Test bad config initialization."""
        with pytest.raises(ValueError):
            loader = RAGQueryLoader(self.graph_data)
        with pytest.raises(ValueError):
            loader = RAGQueryLoader(self.graph_data, config={'d': 'foobar'})
    
    def test_config_propagation(self):
        """Test that config is propagated during initialization."""
        loader = RAGQueryLoader(self.graph_data, config=self.sample_config)
        
        assert loader.feature_store.config == self.sample_config
        assert loader.graph_store.config == self.sample_config
    
    def test_basic_query_without_vector_retriever(self):
        """Test basic query functionality without vector retriever."""
        loader = RAGQueryLoader(self.graph_data, config=self.sample_config)
        
        query = "test query"
        result = loader.query(query)
        
        # Verify result is a Data object
        assert isinstance(result, Data)
        
        # Verify the data has expected attributes
        assert hasattr(result, 'node_idx')
        assert hasattr(result, 'num_nodes')
        assert hasattr(result, 'x')
        assert hasattr(result, 'edge_index')
    
    def test_query_with_vector_retriever(self):
        """Test query functionality with vector retriever."""
        mock_vector_retriever = Mock(spec=VectorRetriever)
        mock_vector_retriever.query.return_value = [
            "retrieved doc 1", "retrieved doc 2"]
        
        loader = RAGQueryLoader(
            self.graph_data,
            vector_retriever=mock_vector_retriever,
            config=self.sample_config
        )
        
        query = "test query"
        result = loader.query(query)
        
        # Verify vector retriever was called
        mock_vector_retriever.query.assert_called_once_with(query)
        
        # Verify result has text_context
        assert hasattr(result, 'text_context')
        assert result.text_context == ["retrieved doc 1", "retrieved doc 2"]
    
    def test_query_with_subgraph_filter(self):
        """Test query functionality with subgraph filter."""
        mock_filter_result = Data()
        mock_filter_result.filtered = True
        
        mock_subgraph_filter = Mock(return_value=mock_filter_result)
        
        loader = RAGQueryLoader(
            self.graph_data,
            subgraph_filter=mock_subgraph_filter,
            config=self.sample_config
        )
        
        query = "test query"
        result = loader.query(query)
        
        # Verify subgraph filter was called
        mock_subgraph_filter.assert_called_once()
        call_args = mock_subgraph_filter.call_args[0]
        assert len(call_args) == 2
        assert call_args[1] == query
        
        # Verify result is the filtered result
        assert result == mock_filter_result
        assert hasattr(result, 'filtered')
        assert result.filtered is True 