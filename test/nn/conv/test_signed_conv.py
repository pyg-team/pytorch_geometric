import torch
from torch_geometric.nn import SignedConv


def test_signed_conv():
    in_channels, out_channels = (16, 32)
    ei_pos = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    ei_neg = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    num_nodes = ei_pos.max().item() + 1
    x = torch.randn((num_nodes, in_channels))

    conv = SignedConv(in_channels, out_channels, first_aggr=True)
    assert conv.__repr__() == 'SignedConv(16, 32, first_aggr=True)'
    x = conv(x, ei_pos, ei_neg)
    assert x.size() == (num_nodes, 2 * out_channels)

    conv = SignedConv(out_channels, out_channels, first_aggr=False)
    assert conv.__repr__() == 'SignedConv(32, 32, first_aggr=False)'
    assert conv(x, ei_pos, ei_neg).size() == (num_nodes, 2 * out_channels)
