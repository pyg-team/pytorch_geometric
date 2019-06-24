import torch
from torch_geometric.nn import FeaStConv


def test_feast_conv():
    in_channels, out_channels = (16, 32)
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    num_nodes = edge_index.max().item() + 1
    x = torch.randn((num_nodes, in_channels))

    conv = FeaStConv(in_channels, out_channels, heads=2)
    assert conv.__repr__() == 'FeaStConv(16, 32, heads=2)'
    assert conv(x, edge_index).size() == (num_nodes, out_channels)
