import torch
from torch.nn import Linear, ReLU, Sequential
from torch_sparse import SparseTensor

from torch_geometric.nn import PointTransformerConv


def test_point_transformer_conv():
    x1 = torch.rand(4, 16)
    x2 = torch.randn(2, 8)
    pos1 = torch.rand(4, 3)
    pos2 = torch.randn(2, 3)
    edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    row, col = edge_index
    adj = SparseTensor(row=row, col=col, sparse_sizes=(4, 4))

    conv = PointTransformerConv(in_channels=16, out_channels=32)
    assert str(conv) == 'PointTransformerConv(16, 32)'

    out = conv(x1, pos1, edge_index)
    assert out.size() == (4, 32)
    assert torch.allclose(conv(x1, pos1, adj.t()), out, atol=1e-6)

    t = '(Tensor, Tensor, Tensor) -> Tensor'
    jit = torch.jit.script(conv.jittable(t))
    assert jit(x1, pos1, edge_index).tolist() == out.tolist()

    t = '(Tensor, Tensor, SparseTensor) -> Tensor'
    jit = torch.jit.script(conv.jittable(t))
    assert torch.allclose(jit(x1, pos1, adj.t()), out, atol=1e-6)

    pos_nn = Sequential(Linear(3, 16), ReLU(), Linear(16, 32))
    attn_nn = Sequential(Linear(32, 32), ReLU(), Linear(32, 32))
    conv = PointTransformerConv(16, 32, pos_nn, attn_nn)

    out = conv(x1, pos1, edge_index)
    assert out.size() == (4, 32)
    assert torch.allclose(conv(x1, pos1, adj.t()), out, atol=1e-6)

    conv = PointTransformerConv((16, 8), 32)
    adj = adj.sparse_resize((4, 2))

    out = conv((x1, x2), (pos1, pos2), edge_index)
    assert out.size() == (2, 32)
    assert torch.allclose(conv((x1, x2), (pos1, pos2), adj.t()), out,
                          atol=1e-6)

    t = '(PairTensor, PairTensor, Tensor) -> Tensor'
    jit = torch.jit.script(conv.jittable(t))
    assert jit((x1, x2), (pos1, pos2), edge_index).tolist() == out.tolist()

    t = '(PairTensor, PairTensor, SparseTensor) -> Tensor'
    jit = torch.jit.script(conv.jittable(t))
    assert torch.allclose(jit((x1, x2), (pos1, pos2), adj.t()), out, atol=1e-6)
