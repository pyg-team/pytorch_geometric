import torch
from torch_sparse import SparseTensor

from torch_geometric.nn import SGConv
from torch_geometric.testing import is_full_test


def test_sg_conv():
    x = torch.randn(4, 16)
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    row, col = edge_index
    value = torch.rand(row.size(0))
    adj2 = SparseTensor(row=row, col=col, value=value, sparse_sizes=(4, 4))
    adj1 = adj2.set_value(None)
    adj3 = adj1.to_torch_sparse_coo_tensor()
    adj4 = adj2.to_torch_sparse_coo_tensor()

    conv = SGConv(16, 32, K=10)
    assert conv.__repr__() == 'SGConv(16, 32, K=10)'
    out1 = conv(x, edge_index)
    assert out1.size() == (4, 32)
    assert torch.allclose(conv(x, adj1.t()), out1, atol=1e-6)
    assert torch.allclose(conv(x, adj3.t()), out1, atol=1e-6)
    out2 = conv(x, edge_index, value)
    assert out2.size() == (4, 32)
    assert torch.allclose(conv(x, adj2.t()), out2, atol=1e-6)
    assert torch.allclose(conv(x, adj4.t()), out2, atol=1e-6)

    if is_full_test():
        t = '(Tensor, Tensor, OptTensor) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert jit(x, edge_index).tolist() == out1.tolist()
        assert jit(x, edge_index, value).tolist() == out2.tolist()

        t = '(Tensor, SparseTensor, OptTensor) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert torch.allclose(jit(x, adj1.t()), out1, atol=1e-6)
        assert torch.allclose(jit(x, adj2.t()), out2, atol=1e-6)

    conv.cached = True
    conv(x, edge_index)
    assert conv(x, edge_index).tolist() == out1.tolist()
    assert torch.allclose(conv(x, adj1.t()), out1, atol=1e-6)
    assert torch.allclose(conv(x, adj3.t()), out1, atol=1e-6)
