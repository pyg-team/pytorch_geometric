import torch
from torch_sparse import SparseTensor

from torch_geometric.nn import CGConv
from torch_geometric.testing import is_full_test


def test_cg_conv():
    x1 = torch.randn(4, 8)
    x2 = torch.randn(2, 16)
    edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    row, col = edge_index
    adj1 = SparseTensor(row=row, col=col, sparse_sizes=(4, 4))
    adj2 = adj1.to_torch_sparse_coo_tensor()

    conv = CGConv(8)
    assert str(conv) == 'CGConv(8, dim=0)'
    out = conv(x1, edge_index)
    assert out.size() == (4, 8)
    assert torch.allclose(conv(x1, adj1.t()), out)
    assert torch.allclose(conv(x1, adj2.t()), out)

    if is_full_test():
        t = '(Tensor, Tensor, OptTensor) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert torch.allclose(jit(x1, edge_index), out, atol=1e-6)

        t = '(Tensor, SparseTensor, OptTensor) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert torch.allclose(jit(x1, adj1.t()), out, atol=1e-6)

    adj1 = adj1.sparse_resize((4, 2))
    adj2 = adj1.to_torch_sparse_coo_tensor()
    conv = CGConv((8, 16))
    assert str(conv) == 'CGConv((8, 16), dim=0)'
    out = conv((x1, x2), edge_index)
    assert out.size() == (2, 16)
    assert torch.allclose(conv((x1, x2), adj1.t()), out, atol=1e-6)
    assert torch.allclose(conv((x1, x2), adj2.t()), out, atol=1e-6)

    if is_full_test():
        t = '(PairTensor, Tensor, OptTensor) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert torch.allclose(jit((x1, x2), edge_index), out, atol=1e-6)

        t = '(PairTensor, SparseTensor, OptTensor) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert torch.allclose(jit((x1, x2), adj1.t()), out, atol=1e-6)

    # Test batch_norm true:
    adj1 = SparseTensor(row=row, col=col, sparse_sizes=(4, 4))
    adj2 = adj1.to_torch_sparse_coo_tensor()
    conv = CGConv(8, batch_norm=True)
    assert str(conv) == 'CGConv(8, dim=0)'
    out = conv(x1, edge_index)
    assert out.size() == (4, 8)
    assert torch.allclose(conv(x1, adj1.t()), out, atol=1e-6)
    assert torch.allclose(conv(x1, adj2.t()), out, atol=1e-6)

    if is_full_test():
        t = '(Tensor, Tensor, OptTensor) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert torch.allclose(jit(x1, edge_index), out, atol=1e-6)


def test_cg_conv_with_edge_features():
    x1 = torch.randn(4, 8)
    x2 = torch.randn(2, 16)
    edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    row, col = edge_index
    value = torch.rand(row.size(0), 3)
    adj = SparseTensor(row=row, col=col, value=value, sparse_sizes=(4, 4))

    conv = CGConv(8, dim=3)
    assert str(conv) == 'CGConv(8, dim=3)'
    out = conv(x1, edge_index, value)
    assert out.size() == (4, 8)
    assert conv(x1, adj.t()).tolist() == out.tolist()

    if is_full_test():
        t = '(Tensor, Tensor, OptTensor) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert jit(x1, edge_index, value).tolist() == out.tolist()

        t = '(Tensor, SparseTensor, OptTensor) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert jit(x1, adj.t()).tolist() == out.tolist()

    adj = adj.sparse_resize((4, 2))
    conv = CGConv((8, 16), dim=3)
    assert str(conv) == 'CGConv((8, 16), dim=3)'
    out = conv((x1, x2), edge_index, value)
    assert out.size() == (2, 16)
    assert conv((x1, x2), adj.t()).tolist() == out.tolist()

    if is_full_test():
        t = '(PairTensor, Tensor, OptTensor) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert jit((x1, x2), edge_index, value).tolist() == out.tolist()

        t = '(PairTensor, SparseTensor, OptTensor) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert jit((x1, x2), adj.t()).tolist() == out.tolist()
