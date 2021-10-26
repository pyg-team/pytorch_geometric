import torch
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch_sparse import SparseTensor
from torch_geometric.nn import GINConv, GINEConv


def test_gin_conv():
    x1 = torch.randn(4, 16)
    x2 = torch.randn(2, 16)
    edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    row, col = edge_index
    adj = SparseTensor(row=row, col=col, sparse_sizes=(4, 4))

    nn = Seq(Lin(16, 32), ReLU(), Lin(32, 32))
    conv = GINConv(nn, train_eps=True)
    assert conv.__repr__() == (
        'GINConv(nn=Sequential(\n'
        '  (0): Linear(in_features=16, out_features=32, bias=True)\n'
        '  (1): ReLU()\n'
        '  (2): Linear(in_features=32, out_features=32, bias=True)\n'
        '))')
    out = conv(x1, edge_index)
    assert out.size() == (4, 32)
    assert conv(x1, edge_index, size=(4, 4)).tolist() == out.tolist()
    assert conv(x1, adj.t()).tolist() == out.tolist()

    t = '(Tensor, Tensor, Size) -> Tensor'
    jit = torch.jit.script(conv.jittable(t))
    assert jit(x1, edge_index).tolist() == out.tolist()
    assert jit(x1, edge_index, size=(4, 4)).tolist() == out.tolist()

    t = '(Tensor, SparseTensor, Size) -> Tensor'
    jit = torch.jit.script(conv.jittable(t))
    assert jit(x1, adj.t()).tolist() == out.tolist()

    adj = adj.sparse_resize((4, 2))
    out1 = conv((x1, x2), edge_index)
    out2 = conv((x1, None), edge_index, (4, 2))
    assert out1.size() == (2, 32)
    assert out2.size() == (2, 32)
    assert conv((x1, x2), edge_index, (4, 2)).tolist() == out1.tolist()
    assert conv((x1, x2), adj.t()).tolist() == out1.tolist()
    assert conv((x1, None), adj.t()).tolist() == out2.tolist()

    t = '(OptPairTensor, Tensor, Size) -> Tensor'
    jit = torch.jit.script(conv.jittable(t))
    assert jit((x1, x2), edge_index).tolist() == out1.tolist()
    assert jit((x1, x2), edge_index, size=(4, 2)).tolist() == out1.tolist()
    assert jit((x1, None), edge_index, size=(4, 2)).tolist() == out2.tolist()

    t = '(OptPairTensor, SparseTensor, Size) -> Tensor'
    jit = torch.jit.script(conv.jittable(t))
    assert jit((x1, x2), adj.t()).tolist() == out1.tolist()
    assert jit((x1, None), adj.t()).tolist() == out2.tolist()


def test_gine_conv():
    x1 = torch.randn(4, 16)
    x2 = torch.randn(2, 16)
    edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    row, col = edge_index
    value = torch.randn(row.size(0), 16)
    adj = SparseTensor(row=row, col=col, value=value, sparse_sizes=(4, 4))

    nn = Seq(Lin(16, 32), ReLU(), Lin(32, 32))
    conv = GINEConv(nn, train_eps=True)
    assert conv.__repr__() == (
        'GINEConv(nn=Sequential(\n'
        '  (0): Linear(in_features=16, out_features=32, bias=True)\n'
        '  (1): ReLU()\n'
        '  (2): Linear(in_features=32, out_features=32, bias=True)\n'
        '))')
    out = conv(x1, edge_index, value)
    assert out.size() == (4, 32)
    assert conv(x1, edge_index, value, size=(4, 4)).tolist() == out.tolist()
    assert conv(x1, adj.t()).tolist() == out.tolist()

    t = '(Tensor, Tensor, OptTensor, Size) -> Tensor'
    jit = torch.jit.script(conv.jittable(t))
    assert jit(x1, edge_index, value).tolist() == out.tolist()
    assert jit(x1, edge_index, value, size=(4, 4)).tolist() == out.tolist()

    t = '(Tensor, SparseTensor, OptTensor, Size) -> Tensor'
    jit = torch.jit.script(conv.jittable(t))
    assert jit(x1, adj.t()).tolist() == out.tolist()

    adj = adj.sparse_resize((4, 2))
    out1 = conv((x1, x2), edge_index, value)
    out2 = conv((x1, None), edge_index, value, (4, 2))
    assert out1.size() == (2, 32)
    assert out2.size() == (2, 32)
    assert conv((x1, x2), edge_index, value, (4, 2)).tolist() == out1.tolist()
    assert conv((x1, x2), adj.t()).tolist() == out1.tolist()
    assert conv((x1, None), adj.t()).tolist() == out2.tolist()

    t = '(OptPairTensor, Tensor, OptTensor, Size) -> Tensor'
    jit = torch.jit.script(conv.jittable(t))
    assert jit((x1, x2), edge_index, value).tolist() == out1.tolist()
    assert jit((x1, x2), edge_index, value,
               size=(4, 2)).tolist() == out1.tolist()
    assert jit((x1, None), edge_index, value,
               size=(4, 2)).tolist() == out2.tolist()

    t = '(OptPairTensor, SparseTensor, OptTensor, Size) -> Tensor'
    jit = torch.jit.script(conv.jittable(t))
    assert jit((x1, x2), adj.t()).tolist() == out1.tolist()
    assert jit((x1, None), adj.t()).tolist() == out2.tolist()


def test_gine_conv_edge_dim():
    x = torch.randn(4, 16)
    edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    edge_attr = torch.randn(edge_index.size(1), 8)

    nn = Seq(Lin(16, 32), ReLU(), Lin(32, 32))
    conv = GINEConv(nn, train_eps=True, edge_dim=8)
    out = conv(x, edge_index, edge_attr)
    assert out.size() == (4, 32)


def test_static_gin_conv():
    x = torch.randn(3, 4, 16)
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])

    nn = Seq(Lin(16, 32), ReLU(), Lin(32, 32))
    conv = GINConv(nn, train_eps=True)
    out = conv(x, edge_index)
    assert out.size() == (3, 4, 32)


def test_static_gine_conv():
    x = torch.randn(3, 4, 16)
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    edge_attr = torch.randn(edge_index.size(1), 16)

    nn = Seq(Lin(16, 32), ReLU(), Lin(32, 32))
    conv = GINEConv(nn, train_eps=True)
    out = conv(x, edge_index, edge_attr)
    assert out.size() == (3, 4, 32)
