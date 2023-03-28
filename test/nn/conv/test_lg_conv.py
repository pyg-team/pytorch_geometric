import torch

from torch_geometric.nn import LGConv
from torch_geometric.testing import is_full_test
from torch_geometric.typing import SparseTensor


def test_lg_conv():
    x = torch.randn(4, 8)
    edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    row, col = edge_index
    value = torch.rand(row.size(0))
    adj2 = SparseTensor(row=row, col=col, value=value, sparse_sizes=(4, 4))
    adj1 = adj2.set_value(None)
    adj3 = adj1.to_torch_sparse_csc_tensor()
    adj4 = adj2.to_torch_sparse_csc_tensor()

    conv = LGConv()
    assert str(conv) == 'LGConv()'
    out1 = conv(x, edge_index)
    assert out1.size() == (4, 8)
    assert torch.allclose(conv(x, adj1.t()), out1)
    assert torch.allclose(conv(x, adj3.t()), out1)
    out2 = conv(x, edge_index, value)
    assert out2.size() == (4, 8)
    assert torch.allclose(conv(x, adj2.t()), out2)
    assert torch.allclose(conv(x, adj4.t()), out2)

    if is_full_test():
        t = '(Tensor, Tensor, OptTensor) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert torch.allclose(jit(x, edge_index), out1)
        assert torch.allclose(jit(x, edge_index, value), out2)

        t = '(Tensor, SparseTensor, OptTensor) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert torch.allclose(jit(x, adj1.t()), out1)
        assert torch.allclose(jit(x, adj2.t()), out2)
