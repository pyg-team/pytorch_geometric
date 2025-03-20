import torch

from torch_geometric.nn.models import ARLinkPredictor


def test_ar_link_predictor():
    model = ARLinkPredictor(in_channels=16, hidden_channels=32, num_layers=2)
    x = torch.randn(4, 16)  # 4 nodes with 16 features each
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])  # 3 edges

    # Test forward pass
    pred = model(x, edge_index)
    assert pred.size(0) == edge_index.size(1)
    assert torch.all(pred >= 0) and torch.all(pred <= 1)

    # Test encode function
    attract_z, repel_z = model.encode(x)
    assert attract_z.size() == (
        4, 16)  # Default attract_ratio=0.5, so half of hidden_channels
    assert repel_z.size() == (4, 16)

    # Test decode function
    raw_scores = model.decode(attract_z, repel_z, edge_index)
    assert raw_scores.size(0) == edge_index.size(1)

    # Test R-fraction calculation
    r_fraction = model.calculate_r_fraction(attract_z, repel_z)
    assert 0 <= r_fraction <= 1


def test_ar_link_predictor_with_custom_ratio():
    # Test with custom attract_ratio
    model = ARLinkPredictor(in_channels=8, hidden_channels=20,
                            attract_ratio=0.7)
    x = torch.randn(5, 8)

    # Check dimensions
    attract_z, repel_z = model.encode(x)
    assert attract_z.size() == (5, 14)  # 70% of 20 = 14
    assert repel_z.size() == (5, 6)  # 30% of 20 = 6
