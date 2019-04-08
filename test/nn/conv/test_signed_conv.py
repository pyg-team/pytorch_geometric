import torch
from torch_geometric.nn import SignedConv


def test_signed_conv():
    in_channels, out_channels = (16, 32)
    pos_ei = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    neg_ei = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    num_nodes = pos_ei.max().item() + 1
    x = torch.randn((num_nodes, in_channels))

    conv = SignedConv(in_channels, out_channels, first_aggr=True)
    assert conv.__repr__() == 'SignedConv(16, 32, first_aggr=True)'
    x = conv(x, pos_ei, neg_ei)
    assert x.size() == (num_nodes, 2 * out_channels)

    conv = SignedConv(out_channels, out_channels, first_aggr=False)
    assert conv.__repr__() == 'SignedConv(32, 32, first_aggr=False)'
    assert conv(x, pos_ei, neg_ei).size() == (num_nodes, 2 * out_channels)
