import torch

from torch_geometric.nn import GCN
from torch_geometric.nn.models.attract_repel import AttractRepel


def test_attract_repel_initialization():
    # Test initialization with string model name
    model1 = AttractRepel('GCN', in_channels=16, hidden_channels=32,
                          out_channels=8, num_layers=2)
    assert model1.attract_dim == 4  # Default attract_ratio=0.5
    assert model1.repel_dim == 4

    # Test initialization with model instance
    base_model = GCN(in_channels=16, hidden_channels=32, out_channels=64,
                     num_layers=2)
    model2 = AttractRepel(base_model, out_channels=10, attract_ratio=0.7)
    assert model2.attract_dim == 7  # 70% of 10
    assert model2.repel_dim == 3

    # Test with uneven split
    model3 = AttractRepel('GCN', in_channels=16, hidden_channels=32,
                          out_channels=9, attract_ratio=0.6, num_layers=2)
    assert model3.attract_dim == 5  # int(9 * 0.6) = 5
    assert model3.repel_dim == 4  # 9 - 5 = 4


def test_attract_repel_forward():
    model = AttractRepel('GCN', in_channels=16, hidden_channels=32,
                         out_channels=8, num_layers=2)

    # Create dummy data
    x = torch.randn(4, 16)  # 4 nodes with 16 features
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])  # 3 edges

    # Test encode function
    attract_z, repel_z = model.encode(x, edge_index)
    assert attract_z.size() == (4, 4)
    assert repel_z.size() == (4, 4)

    # Test forward pass with edge_label_index
    edge_label_index = torch.tensor([[0, 1], [2, 3]])
    scores = model(x, edge_index, edge_label_index=edge_label_index)
    assert scores.size(0) == edge_label_index.size(1)

    # Test forward pass without edge_label_index (return embeddings)
    embeddings = model(x, edge_index)
    assert embeddings.size() == (4, 8)

    # Test score calculation
    manual_scores = model.calculate_scores(attract_z[0:1], attract_z[1:2],
                                           repel_z[0:1], repel_z[1:2])
    assert manual_scores.size(0) == 1


def test_attract_repel_with_different_models():
    # Test with GAT
    gat_model = AttractRepel('GAT', in_channels=16, hidden_channels=32,
                             out_channels=8, num_layers=2)

    # Test with GraphSAGE
    sage_model = AttractRepel('GraphSAGE', in_channels=16, hidden_channels=32,
                              out_channels=8, num_layers=2)

    # Create dummy data
    x = torch.randn(4, 16)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])

    # Verify both models work
    gat_scores = gat_model(x, edge_index, edge_label_index=edge_index)
    sage_scores = sage_model(x, edge_index, edge_label_index=edge_index)

    assert gat_scores.size(0) == edge_index.size(1)
    assert sage_scores.size(0) == edge_index.size(1)


def test_r_fraction_calculation():
    model = AttractRepel('GCN', in_channels=16, hidden_channels=32,
                         out_channels=8, num_layers=2)

    # Create dummy data
    x = torch.randn(4, 16)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])

    # Get embeddings
    attract_z, repel_z = model.encode(x, edge_index)

    # Calculate R-fraction
    r_fraction = model.calculate_r_fraction(attract_z, repel_z)

    # Check it's a valid fraction
    assert 0 <= r_fraction <= 1


def test_custom_model_adaptation():
    # Create a custom model with different output size
    base_model = GCN(in_channels=16, hidden_channels=32, out_channels=20,
                     num_layers=2)

    # Wrap it with AttractRepel with different out_channels
    model = AttractRepel(base_model, out_channels=10)

    # Create dummy data
    x = torch.randn(4, 16)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])

    # Verify the model works and produces correct output sizes
    attract_z, repel_z = model.encode(x, edge_index)
    assert attract_z.size() == (4, 5)  # Half of 10
    assert repel_z.size() == (4, 5)  # Half of 10

    # Check embeddings
    embeddings = model(x, edge_index)
    assert embeddings.size() == (4, 10)
