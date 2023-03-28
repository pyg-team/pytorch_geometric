import torch

import torch_geometric.typing
from torch_geometric.nn import GCN2Conv
from torch_geometric.testing import is_full_test
from torch_geometric.typing import SparseTensor
from torch_geometric.utils import to_torch_csc_tensor


def test_gcn2_conv():
    x = torch.randn(4, 16)
    x_0 = torch.randn(4, 16)
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    value = torch.rand(edge_index.size(1))
    adj1 = to_torch_csc_tensor(edge_index, size=(4, 4))
    adj2 = to_torch_csc_tensor(edge_index, value, size=(4, 4))

    conv = GCN2Conv(16, alpha=0.2)
    assert str(conv) == 'GCN2Conv(16, alpha=0.2, beta=1.0)'
    out1 = conv(x, x_0, edge_index)
    assert out1.size() == (4, 16)
    assert torch.allclose(conv(x, x_0, adj1.t()), out1, atol=1e-6)
    out2 = conv(x, x_0, edge_index, value)
    assert out2.size() == (4, 16)
    assert torch.allclose(conv(x, x_0, adj2.t()), out2, atol=1e-6)

    if torch_geometric.typing.WITH_TORCH_SPARSE:
        adj3 = SparseTensor.from_edge_index(edge_index, sparse_sizes=(4, 4))
        adj4 = SparseTensor.from_edge_index(edge_index, value, (4, 4))
        assert torch.allclose(conv(x, x_0, adj3.t()), out1, atol=1e-6)
        assert torch.allclose(conv(x, x_0, adj4.t()), out2, atol=1e-6)

    if is_full_test():
        t = '(Tensor, Tensor, Tensor, OptTensor) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert jit(x, x_0, edge_index).tolist() == out1.tolist()
        assert jit(x, x_0, edge_index, value).tolist() == out2.tolist()

    if is_full_test() and torch_geometric.typing.WITH_TORCH_SPARSE:
        t = '(Tensor, Tensor, SparseTensor, OptTensor) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert torch.allclose(jit(x, x_0, adj3.t()), out1, atol=1e-6)
        assert torch.allclose(jit(x, x_0, adj4.t()), out2, atol=1e-6)

    conv.cached = True
    conv(x, x_0, edge_index)
    assert conv._cached_edge_index is not None
    assert torch.allclose(conv(x, x_0, edge_index), out1, atol=1e-6)
    assert torch.allclose(conv(x, x_0, adj1.t()), out1, atol=1e-6)

    if torch_geometric.typing.WITH_TORCH_SPARSE:
        conv(x, x_0, adj3.t())
        assert conv._cached_adj_t is not None
        assert torch.allclose(conv(x, x_0, adj3.t()), out1, atol=1e-6)
