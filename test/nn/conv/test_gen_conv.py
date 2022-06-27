import pytest
import torch
from torch_sparse import SparseTensor

from torch_geometric.nn import GENConv
from torch_geometric.testing import is_full_test


@pytest.mark.parametrize('aggr', ['softmax', 'powermean'])
def test_gen_conv(aggr):
    x1 = torch.randn(4, 16)
    x2 = torch.randn(2, 16)
    edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    row, col = edge_index
    value = torch.randn(row.size(0), 16)
    adj1 = SparseTensor(row=row, col=col, sparse_sizes=(4, 4))
    adj2 = SparseTensor(row=row, col=col, value=value, sparse_sizes=(4, 4))

    conv = GENConv(16, 32, aggr)
    assert conv.__repr__() == f'GENConv(16, 32, aggr={aggr})'
    out11 = conv(x1, edge_index)
    assert out11.size() == (4, 32)
    assert torch.allclose(conv(x1, edge_index, size=(4, 4)), out11)
    assert torch.allclose(conv(x1, adj1.t()), out11)

    out12 = conv(x1, edge_index, value)
    assert out12.size() == (4, 32)
    assert torch.allclose(conv(x1, edge_index, value, (4, 4)), out12)
    assert torch.allclose(conv(x1, adj2.t()), out12)

    if is_full_test():
        t = '(Tensor, Tensor, OptTensor, Size) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert torch.allclose(jit(x1, edge_index), out11)
        assert torch.allclose(jit(x1, edge_index, size=(4, 4)), out11)
        assert torch.allclose(jit(x1, edge_index, value), out12)
        assert torch.allclose(jit(x1, edge_index, value, size=(4, 4)), out12)

        t = '(Tensor, SparseTensor, OptTensor, Size) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert torch.allclose(jit(x1, adj1.t()), out11)
        assert torch.allclose(jit(x1, adj2.t()), out12)

    adj1 = adj1.sparse_resize((4, 2))
    adj2 = adj2.sparse_resize((4, 2))

    out21 = conv((x1, x2), edge_index)
    assert out21.size() == (2, 32)
    assert torch.allclose(conv((x1, x2), edge_index, size=(4, 2)), out21)
    assert torch.allclose(conv((x1, x2), adj1.t()), out21)

    out22 = conv((x1, x2), edge_index, value)
    assert out22.size() == (2, 32)
    assert torch.allclose(conv((x1, x2), edge_index, value, (4, 2)), out22)
    assert torch.allclose(conv((x1, x2), adj2.t()), out22)

    if is_full_test():
        t = '(OptPairTensor, Tensor, OptTensor, Size) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert torch.allclose(jit((x1, x2), edge_index), out21)
        assert torch.allclose(jit((x1, x2), edge_index, size=(4, 2)), out21)
        assert torch.allclose(jit((x1, x2), edge_index, value), out22)
        assert torch.allclose(jit((x1, x2), edge_index, value, (4, 2)), out22)

        t = '(OptPairTensor, SparseTensor, OptTensor, Size) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert torch.allclose(jit((x1, x2), adj1.t()), out21)
        assert torch.allclose(jit((x1, x2), adj2.t()), out22)
