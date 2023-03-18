import torch
from torch_sparse import SparseTensor

from torch_geometric.nn import AntiSymmetricConv


def test_antisymmetric_conv():
    x = torch.randn(4, 8)
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    row, col = edge_index
    value = torch.rand(row.size(0))
    adj2 = SparseTensor(row=row, col=col, value=value, sparse_sizes=(4, 4))
    adj1 = adj2.set_value(None)
    adj3 = adj1.to_torch_sparse_coo_tensor()
    adj4 = adj2.to_torch_sparse_coo_tensor()

    conv = AntiSymmetricConv(8)
    assert str(conv) == ('AntiSymmetricConv(8, phi=GCNConv(8, 8), '
                         'num_iters=1, epsilon=0.1, gamma=0.1)')

    out = conv(x, edge_index)
    assert out.size() == (4, 8)
    assert torch.allclose(conv(x, adj1.t()), out, atol=1e-6)
    assert torch.allclose(conv(x, adj3.t()), out, atol=1e-6)

    out = conv(x, edge_index, value)
    assert out.size() == (4, 8)
    assert torch.allclose(conv(x, adj2.t()), out, atol=1e-6)
    assert torch.allclose(conv(x, adj4.t()), out, atol=1e-6)
