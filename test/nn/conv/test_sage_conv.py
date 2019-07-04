import torch
from torch_geometric.nn import SAGEConv


def test_sage_conv():
    in_channels, out_channels = (16, 32)
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    num_nodes = edge_index.max().item() + 1
    x = torch.randn((num_nodes, in_channels))

    conv = SAGEConv(in_channels, out_channels)
    assert conv.__repr__() == 'SAGEConv(16, 32)'
    assert conv(x, edge_index).size() == (num_nodes, out_channels)
    assert conv((x, None), edge_index).size() == (num_nodes, out_channels)
