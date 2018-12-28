import torch
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch_geometric.nn import NNConv


def test_nn_conv():
    in_channels, out_channels = (16, 32)
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    num_nodes = edge_index.max().item() + 1
    x = torch.randn((num_nodes, in_channels))
    pseudo = torch.rand((edge_index.size(1), 3))

    nn = Seq(Lin(3, 32), ReLU(), Lin(32, in_channels * out_channels))
    conv = NNConv(in_channels, out_channels, nn)
    assert conv.__repr__() == 'NNConv(16, 32)'
    assert conv(x, edge_index, pseudo).size() == (num_nodes, out_channels)
