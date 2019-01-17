import torch
from torch_geometric.nn import ARMAConv


def test_arma_conv():
    in_channels, out_channels = (16, 32)
    num_stacks, num_layers = 8, 4
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    num_nodes = edge_index.max().item() + 1
    edge_weight = torch.rand(edge_index.size(1))

    x = torch.randn((num_nodes, in_channels))

    conv = ARMAConv(
        in_channels, out_channels, num_stacks, num_layers, dropout=0.25)
    assert conv.__repr__() == 'ARMAConv(16, 32, num_stacks=8, num_layers=4)'
    assert conv(x, edge_index).size() == (num_nodes, out_channels)
    assert conv(x, edge_index, edge_weight).size() == (num_nodes, out_channels)

    conv = ARMAConv(
        in_channels, out_channels, num_stacks, num_layers, shared_weights=True)
    assert conv(x, edge_index).size() == (num_nodes, out_channels)
    assert conv(x, edge_index, edge_weight).size() == (num_nodes, out_channels)
