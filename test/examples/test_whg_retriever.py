"""Test WHG-Retriever components for basic functionality."""

import sys
from pathlib import Path

import torch

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def test_warehouse_task_head() -> None:
    """Test WarehouseTaskHead basic functionality."""
    from examples.llm.whg_model import WarehouseTaskHead

    # Create task head
    task_head = WarehouseTaskHead(input_dim=128)

    # Test forward pass
    x = torch.randn(10, 128)
    outputs = task_head(x)

    # Verify outputs
    assert 'lineage' in outputs
    assert 'silo' in outputs
    assert 'anomaly' in outputs
    assert outputs['lineage'].shape == (10, 4)
    assert outputs['silo'].shape == (10, 2)
    assert outputs['anomaly'].shape == (10, 2)


def test_warehouse_g_retriever() -> None:
    """Test WarehouseGRetriever basic functionality."""
    from examples.llm.whg_model import WarehouseGRetriever

    # Create retriever (will skip if PyG not available)
    retriever = WarehouseGRetriever(input_dim=64, hidden_dim=32)

    # Test basic functionality
    x = torch.randn(5, 64)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])

    # Test graph encoding
    embeddings = retriever.encode_graph(x, edge_index)
    assert embeddings.shape == (5, 32)

    # Test node retrieval
    node_metadata = {
        i: {
            'type': f'table_{i}',
            'layer': 'unknown'
        }
        for i in range(5)
    }
    relevant_nodes = retriever.retrieve_relevant_nodes('test question', x,
                                                       node_metadata)
    assert isinstance(relevant_nodes, list)
    assert len(relevant_nodes) <= 10


def test_warehouse_conversation_system() -> None:
    """Test WarehouseConversationSystem basic functionality."""
    from examples.llm.whg_demo import WarehouseConversationSystem

    # Create simple test data
    x = torch.randn(5, 64)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])

    # Create conversation system
    warehouse = WarehouseConversationSystem(x, edge_index)

    # Test ask method
    response = warehouse.ask("What is the structure?")
    assert isinstance(response, str)
    assert len(response) > 0
