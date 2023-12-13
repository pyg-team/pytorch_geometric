import torch
from torch.nn import Linear as Lin
from torch.nn import ReLU
from torch.nn import Sequential as Seq

from torch_geometric.nn import EDPConv, MultiChannelConv


def test_multi_channel_conv():
    x = torch.randn(4, 16)
    edge_indices = [
        torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]]),
        torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    ]
    edge_weights = [
        torch.tensor([0.1, 1.1, 2.1, 3.1]),
        torch.tensor([0.2, 1.2, 2.2, 3.2])
    ]

    nn = nn = Seq(Lin(32, 64), ReLU(), Lin(64, 16))
    mc_conv = MultiChannelConv(nn, eps=0, train_eps=True)
    assert str(mc_conv) == (
        'MultiChannelConv(nn=Sequential(\n'
        '  (0): Linear(in_features=32, out_features=64, bias=True)\n'
        '  (1): ReLU()\n'
        '  (2): Linear(in_features=64, out_features=16, bias=True)\n'
        '))')
    out = mc_conv(x, edge_indices, edge_weights)
    assert out.size() == (4, 16)
    assert torch.allclose(mc_conv(x, edge_indices, edge_weights, size=(4, 4)),
                          out, atol=1e-6)


def test_edp_conv():
    x = torch.randn(4, 16)
    edge_indices = [
        torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]]),
        torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    ]
    edge_weights = [
        torch.tensor([0.1, 1.1, 2.1, 3.1]),
        torch.tensor([0.2, 1.2, 2.2, 3.2])
    ]

    nn_node = Seq(Lin(32, 64), ReLU(), Lin(64, 16))
    nn_edge = Seq(Lin(34, 32), ReLU(), Lin(32, 5))
    edp_conv = EDPConv(nn_node, nn_edge, train_eps=True)
    out, edge_indices, edge_weights = edp_conv(x, edge_indices, edge_weights)
    assert out.size() == (4, 16)
    assert len(edge_indices) == len(edge_indices)
    assert len(edge_indices) == 5
    assert edge_indices[0].size() == (2, 16)
    assert edge_weights[0].size() == (16, )
