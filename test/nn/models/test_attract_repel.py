import torch

from torch_geometric.nn import GCN
from torch_geometric.nn.models import AttractRepel


def test_attract_repel_initialization():
    # Test initialization with string model name
    model1 = AttractRepel('GCN', in_channels=16, hidden_channels=32,
                          out_channels=8)
    assert model1.attract_dim == 4  # Default attract_ratio=0.5
    assert model1.repel_dim == 4

    # Test initialization with model instance
    base_model = GCN(in_channels=16, hidden_channels=32, out_channels=64)
    model2 = AttractRepel(base_model, out_channels=10, attract_ratio=0.7)
    assert model2.attract_dim == 7  # 70% of 10
    assert model2.repel_dim == 3


def test_attract_repel_forward():
    model = AttractRepel('GCN', in_channels=16, hidden_channels=32,
                         out_channels=8)

    # Create dummy data
    x = torch.randn(4, 16)  # 4 nodes with 16 features
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])  # 3 edges

    # Test encode function
    attract_z, repel_z = model.encode(x, edge_index)
    assert attract_z.size() == (4, 4)
    assert repel_z.size() == (4, 4)

    # Test forward pass with edge_index
    scores = model(x, edge_index, edge_index=edge_index)
    assert scores.size(0) == edge_index.size(1)

    # Test forward pass without edge_index
    embeddings = model(x, edge_index)
    assert embeddings.size() == (4, 8)


def test_attract_repel_with_different_models():
    # Test with GAT
    gat_model = AttractRepel('GAT', in_channels=16, hidden_channels=32,
                             out_channels=8)

    # Test with GraphSAGE
    sage_model = AttractRepel('GraphSAGE', in_channels=16, hidden_channels=32,
                              out_channels=8)

    # Create dummy data
    x = torch.randn(4, 16)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])

    # Verify both models work
    gat_scores = gat_model(x, edge_index, edge_index=edge_index)
    sage_scores = sage_model(x, edge_index, edge_index=edge_index)

    assert gat_scores.size(0) == edge_index.size(1)
    assert sage_scores.size(0) == edge_index.size(1)


def test_r_fraction_calculation():
    model = AttractRepel('GCN', in_channels=16, hidden_channels=32,
                         out_channels=8)

    # Create dummy data
    x = torch.randn(4, 16)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])

    # Get embeddings
    attract_z, repel_z = model.encode(x, edge_index)

    # Calculate R-fraction
    r_fraction = model.calculate_r_fraction(attract_z, repel_z)

    # Check it's a valid fraction
    assert 0 <= r_fraction <= 1
