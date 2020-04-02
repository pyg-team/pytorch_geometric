import torch
from torch_geometric.nn import GravNetConv


def test_gravnet_conv_conv():
    in_channels, out_channels = (16, 32)
    space_dimensions_s=4
    propagate_dimensions_flr=8
    k_neighbors=12
                 
    num_nodes = 20
    x = torch.randn((num_nodes, in_channels))

    conv = GravNetConv(in_channels,space_dimensions_s,
                       propagate_dimensions_flr,k_neighbors,
                       out_channels)
    assert conv(x).size() == (num_nodes, out_channels)

