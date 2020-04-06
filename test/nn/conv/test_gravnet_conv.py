import torch
from torch_geometric.nn import GravNetConv


def test_gravnet_conv():
    num_nodes, in_channels, out_channels = 20, 16, 32
    x = torch.randn((num_nodes, in_channels))

    conv = GravNetConv(in_channels, out_channels, space_dimensions=4,
                       propagate_dimensions=8, k=12)
    assert conv.__repr__() == 'GravNetConv(16, 32, k=12)'
    assert conv(x).size() == (num_nodes, out_channels)
