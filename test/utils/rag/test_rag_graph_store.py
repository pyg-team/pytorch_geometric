import torch
from unittest.mock import Mock, patch

from torch_geometric.data import FeatureStore
from torch_geometric.sampler import (
    BidirectionalNeighborSampler,
    SamplerOutput,
)
from torch_geometric.utils.rag.graph_store import NeighborSamplingRAGGraphStore

def setup_test_fixtures():
    """Set up test fixtures."""
    feature_store = Mock(spec=FeatureStore)
    config = {"num_neighbors": [10, 5]}
    return feature_store, config


def test_sample_subgraph_with_valid_tensor_input():
    """Test sample_subgraph with valid tensor input."""
    # Create graph store and set config
    feature_store, config = setup_test_fixtures()
    graph_store = NeighborSamplingRAGGraphStore(
        feature_store=feature_store,
        replace=True,
        disjoint=False
    )
    graph_store.config = config
    
    # Create mock sampler and its output
    mock_sampler = Mock(spec=BidirectionalNeighborSampler)
    expected_output = SamplerOutput(
        node=torch.tensor([0, 1, 2, 3]),
        row=torch.tensor([0, 1, 2]),
        col=torch.tensor([1, 2, 3]),
        edge=torch.tensor([0, 1, 2]),
        batch=None,
        num_sampled_nodes=[2, 2],
        num_sampled_edges=[3]
    )
    mock_sampler.sample_from_nodes.return_value = expected_output

    # Initially sampler should not be initialized
    assert not graph_store._sampler_is_initialized
    
    # Mock the _init_sampler method to set our mock sampler
    with patch.object(graph_store, '_init_sampler') as mock_init:
        def set_sampler():
            graph_store.sampler = mock_sampler
            graph_store._sampler_is_initialized = True
        mock_init.side_effect = set_sampler
        
        # Test input
        seed_nodes = torch.tensor([0, 1, 2, 2, 1])  # includes duplicates
        result = graph_store.sample_subgraph(seed_nodes)
        
        # Verify sampler was initialized
        mock_init.assert_called_once()
        
        # Verify sample_from_nodes was called with correct input
        mock_sampler.sample_from_nodes.assert_called_once()
        assert result == expected_output