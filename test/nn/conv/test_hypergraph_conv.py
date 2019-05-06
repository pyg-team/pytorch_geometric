import torch
from torch_geometric.nn import HypergraphConv


def test_hypergraph_conv():
    in_channels, out_channels = (16, 32)
    hyper_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    hyper_weight = torch.tensor([1, 0.5, 0.3, 0.7])
    num_nodes = hyper_index.max().item() + 1
    x = torch.randn((num_nodes, in_channels))

    conv = HypergraphConv(in_channels, out_channels)
    assert conv.__repr__() == 'HypergraphConv(16, 32)'
    out = conv(x, hyper_index, hyper_weight)
    assert out.size() == (num_nodes, out_channels)
