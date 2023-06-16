import torch

from torch_geometric.nn import DirGNNConv, SAGEConv


def test_dir_gnn_conv():
    x = torch.randn(4, 16)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])

    conv = DirGNNConv(SAGEConv(16, 32))
    assert str(conv) == 'DirGNNConv(SAGEConv(16, 32, aggr=mean), alpha=0.5)'

    out = conv(x, edge_index)
    assert out.size() == (4, 32)


def test_static_dir_gnn_conv():
    x = torch.randn(3, 4, 16)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])

    conv = DirGNNConv(SAGEConv(16, 32))

    out = conv(x, edge_index)
    assert out.size() == (3, 4, 32)
