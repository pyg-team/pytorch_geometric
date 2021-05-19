import torch

from torch_geometric.nn.conv.cayley_conv import CayleyConv


def test_cayley_conv():
    in_channels, out_channels = (16, 32)
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    num_nodes = edge_index.max().item() + 1
    edge_weight = torch.rand(edge_index.size(1))
    x = torch.randn((num_nodes, in_channels))

    conv = CayleyConv(in_channels, out_channels, r=3)
    assert conv.__repr__() == 'CayleyConv(16, 32, r=3, normalization=sym)'
    out1 = conv(x, edge_index)
    assert out1.size() == (num_nodes, out_channels)
    out2 = conv(x, edge_index, edge_weight)
    assert out2.size() == (num_nodes, out_channels)
    out3 = conv(x, edge_index, edge_weight)
    assert out3.size() == (num_nodes, out_channels)
