import torch

from torch_geometric.nn import MFConv
from torch_geometric.testing import is_full_test
from torch_geometric.typing import SparseTensor


def test_mf_conv():
    x1 = torch.randn(4, 8)
    x2 = torch.randn(2, 16)
    edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    row, col = edge_index
    adj = SparseTensor(row=row, col=col, sparse_sizes=(4, 4))

    conv = MFConv(8, 32)
    assert str(conv) == 'MFConv(8, 32)'
    out = conv(x1, edge_index)
    assert out.size() == (4, 32)
    assert conv(x1, edge_index, size=(4, 4)).tolist() == out.tolist()
    assert conv(x1, adj.t()).tolist() == out.tolist()

    if is_full_test():
        t = '(Tensor, Tensor, Size) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert jit(x1, edge_index).tolist() == out.tolist()
        assert jit(x1, edge_index, size=(4, 4)).tolist() == out.tolist()

        t = '(Tensor, SparseTensor, Size) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert jit(x1, adj.t()).tolist() == out.tolist()

    adj = adj.sparse_resize((4, 2))
    conv = MFConv((8, 16), 32)
    assert str(conv) == 'MFConv((8, 16), 32)'
    out1 = conv((x1, x2), edge_index)
    out2 = conv((x1, None), edge_index, (4, 2))
    assert out1.size() == (2, 32)
    assert out2.size() == (2, 32)
    assert conv((x1, x2), edge_index, (4, 2)).tolist() == out1.tolist()
    assert conv((x1, x2), adj.t()).tolist() == out1.tolist()
    assert conv((x1, None), adj.t()).tolist() == out2.tolist()

    if is_full_test():
        t = '(OptPairTensor, Tensor, Size) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert jit((x1, x2), edge_index).tolist() == out1.tolist()
        assert jit((x1, x2), edge_index, size=(4, 2)).tolist() == out1.tolist()
        assert jit((x1, None), edge_index,
                   size=(4, 2)).tolist() == out2.tolist()

        t = '(OptPairTensor, SparseTensor, Size) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert jit((x1, x2), adj.t()).tolist() == out1.tolist()
        assert jit((x1, None), adj.t()).tolist() == out2.tolist()
