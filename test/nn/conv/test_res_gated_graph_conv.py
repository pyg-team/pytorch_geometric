import torch
from torch_sparse import SparseTensor

from torch_geometric.nn import ResGatedGraphConv
from torch_geometric.testing import is_full_test


def test_res_gated_graph_conv():
    x1 = torch.randn(4, 8)
    x2 = torch.randn(2, 32)
    edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    row, col = edge_index
    adj1 = SparseTensor(row=row, col=col, sparse_sizes=(4, 4))
    adj2 = adj1.to_torch_sparse_csc_tensor()

    conv = ResGatedGraphConv(8, 32)
    assert str(conv) == 'ResGatedGraphConv(8, 32)'
    out = conv(x1, edge_index)
    assert out.size() == (4, 32)
    assert torch.allclose(conv(x1, adj1.t()), out, atol=1e-6)
    assert torch.allclose(conv(x1, adj2.t()), out, atol=1e-6)

    if is_full_test():
        t = '(Tensor, Tensor) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert torch.allclose(jit(x1, edge_index), out, atol=1e-6)

        t = '(Tensor, SparseTensor) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert torch.allclose(jit(x1, adj1.t()), out, atol=1e-6)

    adj1 = adj1.sparse_resize((4, 2))
    adj2 = adj1.to_torch_sparse_csc_tensor()
    conv = ResGatedGraphConv((8, 32), 32)
    assert str(conv) == 'ResGatedGraphConv((8, 32), 32)'
    out = conv((x1, x2), edge_index)
    assert out.size() == (2, 32)
    assert torch.allclose(conv((x1, x2), adj1.t()), out, atol=1e-6)
    assert torch.allclose(conv((x1, x2), adj2.t()), out, atol=1e-6)

    if is_full_test():
        t = '(PairTensor, Tensor) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert torch.allclose(jit((x1, x2), edge_index), out, atol=1e-6)

        t = '(PairTensor, SparseTensor) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert torch.allclose(jit((x1, x2), adj1.t()), out, atol=1e-6)
