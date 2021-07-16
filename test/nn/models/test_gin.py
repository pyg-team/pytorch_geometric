import torch
from torch_geometric.nn.models.gin import GIN, GINWithJK, GINE, GINEWithJK


def test_gin():
    model = GIN(in_channels=8, hidden_channels=16, num_layers=3)
    out = 'GIN(16, 8, num_layers=3)'
    assert model.__repr__() == out

    x = torch.randn(3, 8)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])

    out = model(x, edge_index)
    assert out.size() == (3, 16)

def test_ginwithjk():
    model = GINWithJK(in_channels=8, hidden_channels=16, num_layers=3)
    out = 'GINWithJK(16, 8, num_layers=3)'
    assert model.__repr__() == out

    x = torch.randn(3, 8)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])

    out = model(x, edge_index)
    assert out.size() == (3, 16)

def test_gine():
    model = GINE(in_channels=8, hidden_channels=16, num_layers=3)
    out = 'GINE(16, 8, num_layers=3)'
    assert model.__repr__() == out

    x = torch.randn(3, 8)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])

    out = model(x, edge_index)
    assert out.size() == (3, 16)

def test_ginewithjk():
    model = GINEWithJK(in_channels=8, hidden_channels=16, num_layers=3)
    out = 'GINEWithJK(16, 8, num_layers=3)'
    assert model.__repr__() == out

    x = torch.randn(3, 8)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])

    out = model(x, edge_index)
    assert out.size() == (3, 16)
