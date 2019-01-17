import torch
from torch_geometric.nn import ARMAConv


def test_arma_conv():
    in_channels, out_channels, initial_channels = (16, 32, 64)
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    num_nodes = edge_index.max().item() + 1
    edge_weight = torch.rand(edge_index.size(1))

    x = torch.randn((num_nodes, in_channels))
    x_init = torch.randn((num_nodes, initial_channels))

    conv = ARMAConv(in_channels, out_channels, initial_channels, dropout=0.25)
    assert conv.__repr__() == 'ARMAConv(16, 32, num_stacks=1)'
    assert conv(x, x_init, edge_index).size() == (num_nodes, out_channels)
    size = conv(x, x_init, edge_index, edge_weight).size()
    assert size == (num_nodes, out_channels)

    num_stacks = 8
    x = torch.randn((num_nodes, num_stacks, in_channels))
    x_init = torch.randn((num_nodes, initial_channels))

    conv = ARMAConv(in_channels, out_channels, initial_channels, num_stacks)
    size = conv(x, x_init, edge_index).size()
    assert size == (num_nodes, num_stacks, out_channels)
    size = conv(x, x_init, edge_index, edge_weight).size()
    assert size == (num_nodes, num_stacks, out_channels)
