import torch
from torch_sparse import SparseTensor
from torch_geometric.nn import AntiSymmetricConv


def test_antisymmetric_conv():
    x1 = torch.randn(4, 8)
    edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    row, col = edge_index
    value = torch.rand(row.size(0))
    adj = SparseTensor(row=row, col=col, value=value, sparse_sizes=(4, 4))

    conv = AntiSymmetricConv(8)
    assert conv.__repr__() == (
        'AntiSymmetricConv(in_channels=8, phi=GCNConv(8, 8), num_iters=1, '
        'epsilon=0.1, gamma=0.1, activ_fun=tanh, bias=True)')
    
    out1 = conv(x1, edge_index, value)
    assert out1.size() == (4, 8)
    assert conv(x1, edge_index, value).tolist() == out1.tolist()
    assert torch.allclose(conv(x1, edge_index, value), out1)
    
    out2 = conv(x1, adj.t())
    assert conv(x1, adj.t()).tolist() == out2.tolist()
    assert torch.allclose(conv(x1, adj.t()), out2)

    out3 = conv(x1, edge_index)
    assert out3.size() == (4, 8)
    assert conv(x1, edge_index).tolist() == out3.tolist()
    assert torch.allclose(conv(x1, edge_index), out3)
