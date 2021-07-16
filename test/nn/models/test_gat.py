import torch
from torch_geometric.nn.models.gat import GAT, GATv2, GATWithJK, GATv2WithJK


def test_gat():
    model = GAT(in_channels=8, hidden_channels=16, num_layers=3)
    out = 'GAT(16, 8, num_layers=3)'
    assert model.__repr__() == out

    x = torch.randn(3, 8)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])

    out = model(x, edge_index)
    assert out.size() == (3, 16)

def test_gatv2():
    model = GATv2(in_channels=8, hidden_channels=16, num_layers=3)
    out = 'GATv2(16, 8, num_layers=3)'
    assert model.__repr__() == out

    x = torch.randn(3, 8)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])

    out = model(x, edge_index)
    assert out.size() == (3, 16)

def test_gatwithjk():
    model = GATWithJK(in_channels=8, hidden_channels=16, num_layers=3)
    out = 'GATWithJK(16, 8, num_layers=3)'
    assert model.__repr__() == out

    x = torch.randn(3, 8)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])

    out = model(x, edge_index)
    assert out.size() == (3, 16)

def test_gatv2withjk():
    model = GATv2WithJK(in_channels=8, hidden_channels=16, num_layers=3)
    out = 'GATv2(16, 8, num_layers=3)'
    assert model.__repr__() == out

    x = torch.randn(3, 8)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])

    out = model(x, edge_index)
    assert out.size() == (3, 16)