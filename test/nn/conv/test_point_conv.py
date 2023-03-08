import torch
from torch.nn import Linear as Lin
from torch.nn import ReLU
from torch.nn import Sequential as Seq
from torch_sparse import SparseTensor

from torch_geometric.nn import PointNetConv
from torch_geometric.testing import is_full_test


def test_point_net_conv():
    x1 = torch.randn(4, 16)
    pos1 = torch.randn(4, 3)
    pos2 = torch.randn(2, 3)
    edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    row, col = edge_index
    adj = SparseTensor(row=row, col=col, sparse_sizes=(4, 4))

    local_nn = Seq(Lin(16 + 3, 32), ReLU(), Lin(32, 32))
    global_nn = Seq(Lin(32, 32))
    conv = PointNetConv(local_nn, global_nn)
    assert str(conv) == (
        'PointNetConv(local_nn=Sequential(\n'
        '  (0): Linear(in_features=19, out_features=32, bias=True)\n'
        '  (1): ReLU()\n'
        '  (2): Linear(in_features=32, out_features=32, bias=True)\n'
        '), global_nn=Sequential(\n'
        '  (0): Linear(in_features=32, out_features=32, bias=True)\n'
        '))')
    out = conv(x1, pos1, edge_index)
    assert out.size() == (4, 32)
    assert torch.allclose(conv(x1, pos1, adj.t()), out, atol=1e-6)

    if is_full_test():
        t = '(OptTensor, Tensor, Tensor) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert jit(x1, pos1, edge_index).tolist() == out.tolist()

        t = '(OptTensor, Tensor, SparseTensor) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert torch.allclose(jit(x1, pos1, adj.t()), out, atol=1e-6)

    adj = adj.sparse_resize((4, 2))
    out = conv(x1, (pos1, pos2), edge_index)
    assert out.size() == (2, 32)
    assert conv((x1, None), (pos1, pos2), edge_index).tolist() == out.tolist()
    assert torch.allclose(conv(x1, (pos1, pos2), adj.t()), out, atol=1e-6)
    assert torch.allclose(conv((x1, None), (pos1, pos2), adj.t()), out,
                          atol=1e-6)

    if is_full_test():
        t = '(PairOptTensor, PairTensor, Tensor) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert jit((x1, None), (pos1, pos2),
                   edge_index).tolist() == out.tolist()

        t = '(PairOptTensor, PairTensor, SparseTensor) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert torch.allclose(jit((x1, None), (pos1, pos2), adj.t()), out,
                              atol=1e-6)
