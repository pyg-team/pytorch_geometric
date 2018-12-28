import torch
from torch_geometric.nn import AGNNConv


def test_agnn_conv():
    in_channels = 16
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    num_nodes = edge_index.max().item() + 1
    x = torch.randn((num_nodes, in_channels))

    conv = AGNNConv(requires_grad=True)
    assert conv.__repr__() == 'AGNNConv()'
    assert conv(x, edge_index).size() == (num_nodes, in_channels)
    conv = AGNNConv(requires_grad=False)
    assert conv(x, edge_index).size() == (num_nodes, in_channels)
