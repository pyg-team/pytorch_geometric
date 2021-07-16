import torch
from torch_geometric.nn.models.graph_sage import GraphSAGE, GraphSAGEWithJK


def test_graph_sage():
    model = GraphSAGE(in_channels=8, hidden_channels=16, num_layers=3)
    out = 'GraphSAGE(16, 8, num_layers=3)'
    assert model.__repr__() == out

    x = torch.randn(3, 8)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])

    out = model(x, edge_index)
    assert out.size() == (3, 16)

def test_graph_sagewithjk():
    model = GraphSAGEWithJK(in_channels=8, hidden_channels=16, num_layers=3)
    out = 'GraphSAGEWithJK(16, 8, num_layers=3)'
    assert model.__repr__() == out

    x = torch.randn(3, 8)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])

    out = model(x, edge_index)
    assert out.size() == (3, 16)
