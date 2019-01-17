import torch
from torch_geometric.nn import ChebConv


def test_cheb_conv():
    in_channels, out_channels = (16, 32)
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    num_nodes = edge_index.max().item() + 1
    edge_weight = torch.rand(edge_index.size(1))
    x = torch.randn((num_nodes, in_channels))

    conv = ChebConv(in_channels, out_channels, K=3)
    assert conv.__repr__() == 'ChebConv(16, 32, K=3)'
    assert conv(x, edge_index).size() == (num_nodes, out_channels)
    assert conv(x, edge_index, edge_weight).size() == (num_nodes, out_channels)
