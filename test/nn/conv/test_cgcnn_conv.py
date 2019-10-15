import torch
from torch_geometric.nn import CGCNNConv


def test_cgcnn_conv():
    node_dim, edge_dim = (16, 32)
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    num_nodes = edge_index.max().item() + 1
    x = torch.randn((num_nodes, node_dim))
    pseudo = torch.rand((edge_index.size(1), edge_dim))

    conv = CGCNNConv(node_dim, edge_dim)
    assert conv.__repr__() == 'CGCNNConv(16, 16)'
    assert conv(x, edge_index, pseudo).size() == (num_nodes, node_dim)
