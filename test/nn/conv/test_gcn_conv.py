import torch
from torch_geometric.nn import GCNConv


def test_gcn_conv():
    in_channels, out_channels = (16, 32)
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    num_nodes = edge_index.max().item() + 1
    edge_weight = torch.rand(edge_index.size(1))
    x = torch.randn((num_nodes, in_channels))

    conv = GCNConv(in_channels, out_channels)
    assert conv.__repr__() == 'GCNConv(16, 32)'
    assert conv(x, edge_index).size() == (num_nodes, out_channels)
    assert conv(x, edge_index, edge_weight).size() == (num_nodes, out_channels)

    x = torch.sparse_coo_tensor(torch.tensor([[0, 0], [0, 1]]),
                                torch.Tensor([1, 1]),
                                torch.Size([num_nodes, in_channels]))
    conv(x, edge_index).size() == (num_nodes, out_channels)

    conv = GCNConv(in_channels, out_channels, cached=True)
    conv(x, edge_index).size() == (num_nodes, out_channels)
    conv(x, edge_index).size() == (num_nodes, out_channels)
