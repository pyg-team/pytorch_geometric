import torch
from torch_sparse import SparseTensor

from torch_geometric.nn import PEGConv
from torch_geometric.testing import is_full_test


def test_peg_conv():
    x = torch.randn(4, 16)  #node_feature
    pos_encoding = torch.randn(4, 16)
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    row, col = edge_index
    value = torch.rand(row.size(0))
    adj2 = SparseTensor(row=row, col=col, value=value, sparse_sizes=(4, 4))
    adj1 = adj2.set_value(None)

    conv = PEGConv(in_channels=16, out_channels=8)
    assert conv.__repr__() == 'PEGConv(16,8)'
    out1 = conv(x, pos_encoding, edge_index)
    assert out1.size() == (4, 8)
    assert torch.allclose(conv(x, pos_encoding, adj1.t()), out1, atol=1e-6)
    out2 = conv(x, pos_encoding, edge_index, value)
    assert out2.size() == (4, 8)
    assert torch.allclose(conv(x, pos_encoding, adj2.t()), out2, atol=1e-6)

    if is_full_test():
        t = '(Tensor, Tensor, Tensor, OptTensor) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert jit(x, pos_encoding, edge_index).tolist() == out1.tolist()
        assert jit(x, pos_encoding, edge_index,
                   value).tolist() == out2.tolist()

        t = '(Tensor, Tensor, SparseTensor, OptTensor) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert torch.allclose(jit(x, pos_encoding, adj1.t()), out1, atol=1e-6)
        assert torch.allclose(jit(x, pos_encoding, adj2.t()), out2, atol=1e-6)

    conv.cached = True
    conv(x, pos_encoding, edge_index)
    assert conv(x, pos_encoding, edge_index).tolist() == out1.tolist()
    conv(x, pos_encoding, adj1.t())
    assert torch.allclose(conv(x, pos_encoding, adj1.t()), out1, atol=1e-6)
