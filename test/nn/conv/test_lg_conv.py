import torch

import torch_geometric.typing
from torch_geometric.nn import LGConv
from torch_geometric.testing import is_full_test
from torch_geometric.typing import SparseTensor
from torch_geometric.utils import to_torch_csc_tensor


def test_lg_conv():
    x = torch.randn(4, 8)
    edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    value = torch.rand(edge_index.size(1))
    adj1 = to_torch_csc_tensor(edge_index, size=(4, 4))
    adj2 = to_torch_csc_tensor(edge_index, value, size=(4, 4))

    conv = LGConv()
    assert str(conv) == 'LGConv()'
    out1 = conv(x, edge_index)
    assert out1.size() == (4, 8)
    assert torch.allclose(conv(x, adj1.t()), out1)

    out2 = conv(x, edge_index, value)
    assert out2.size() == (4, 8)
    assert torch.allclose(conv(x, adj2.t()), out2)

    if torch_geometric.typing.WITH_TORCH_SPARSE:
        adj3 = SparseTensor.from_edge_index(edge_index, sparse_sizes=(4, 4))
        adj4 = SparseTensor.from_edge_index(edge_index, value, (4, 4))
        assert torch.allclose(conv(x, adj3.t()), out1)
        assert torch.allclose(conv(x, adj4.t()), out2)

    if is_full_test():
        t = '(Tensor, Tensor, OptTensor) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert torch.allclose(jit(x, edge_index), out1)
        assert torch.allclose(jit(x, edge_index, value), out2)

    if is_full_test() and torch_geometric.typing.WITH_TORCH_SPARSE:
        t = '(Tensor, SparseTensor, OptTensor) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert torch.allclose(jit(x, adj3.t()), out1)
        assert torch.allclose(jit(x, adj4.t()), out2)
