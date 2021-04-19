import torch
from torch_geometric.nn import EGConv
from torch_sparse import SparseTensor


def test_egconv():
    x = torch.randn(4, 16)
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    row, col = edge_index
    adj1 = SparseTensor(row=row, col=col, sparse_sizes=(4, 4))

    conv = EGConv(16, 32)
    assert conv.__repr__() == 'EGConv(16, 32, [\'symnorm\'])'
    out1 = conv(x, edge_index)
    assert out1.size() == (4, 32)
    assert torch.allclose(conv(x, adj1.t()), out1, atol=1e-6)

    conv.cached = True
    conv(x, edge_index)
    assert conv(x, edge_index).tolist() == out1.tolist()
    conv(x, adj1.t())
    assert torch.allclose(conv(x, adj1.t()), out1, atol=1e-6)

    t = '(Tensor, Tensor) -> Tensor'
    jit = torch.jit.script(conv.jittable(t))
    assert jit(x, edge_index).tolist() == out1.tolist()

    t = '(Tensor, SparseTensor) -> Tensor'
    jit = torch.jit.script(conv.jittable(t))
    assert torch.allclose(jit(x, adj1.t()), out1, atol=1e-6)


def test_egconv_multi():
    x = torch.randn(4, 16)
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    row, col = edge_index
    adj1 = SparseTensor(row=row, col=col, sparse_sizes=(4, 4))

    conv = EGConv(16, 32, aggrs=["max", "min"])
    assert conv.__repr__() == 'EGConv(16, 32, [\'max\', \'min\'])'
    out1 = conv(x, edge_index)
    assert out1.size() == (4, 32)
    assert torch.allclose(conv(x, adj1.t()), out1, atol=1e-6)

    conv.cached = True
    conv(x, edge_index)
    assert conv(x, edge_index).tolist() == out1.tolist()
    conv(x, adj1.t())
    assert torch.allclose(conv(x, adj1.t()), out1, atol=1e-6)

    t = '(Tensor, Tensor) -> Tensor'
    jit = torch.jit.script(conv.jittable(t))
    assert jit(x, edge_index).tolist() == out1.tolist()

    t = '(Tensor, SparseTensor) -> Tensor'
    jit = torch.jit.script(conv.jittable(t))
    assert torch.allclose(jit(x, adj1.t()), out1, atol=1e-6)


def test_egconv_with_sparse_input_feature():
    x = torch.sparse_coo_tensor(indices=torch.tensor([[0, 0], [0, 1]]),
                                values=torch.tensor([1., 1.]),
                                size=torch.Size([4, 16]))
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])

    conv = EGConv(16, 32)
    assert conv(x, edge_index).size() == (4, 32)
