import torch

import torch_geometric.typing
from torch_geometric.nn import EGConv
from torch_geometric.testing import is_full_test
from torch_geometric.typing import SparseTensor
from torch_geometric.utils import to_torch_csc_tensor


def test_eg_conv():
    x = torch.randn(4, 16)
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    adj1 = to_torch_csc_tensor(edge_index, size=(4, 4))

    conv = EGConv(16, 32)
    assert str(conv) == "EGConv(16, 32, aggregators=['symnorm'])"
    out = conv(x, edge_index)
    assert out.size() == (4, 32)
    assert torch.allclose(conv(x, adj1.t()), out, atol=1e-6)

    if torch_geometric.typing.WITH_TORCH_SPARSE:
        adj2 = SparseTensor.from_edge_index(edge_index, sparse_sizes=(4, 4))
        assert torch.allclose(conv(x, adj2.t()), out, atol=1e-6)

    conv.cached = True
    assert torch.allclose(conv(x, edge_index), out, atol=1e-6)
    assert conv._cached_edge_index is not None
    assert torch.allclose(conv(x, edge_index), out, atol=1e-6)
    assert torch.allclose(conv(x, adj1.t()), out, atol=1e-6)

    if torch_geometric.typing.WITH_TORCH_SPARSE:
        assert torch.allclose(conv(x, adj2.t()), out, atol=1e-6)
        assert conv._cached_adj_t is not None
        assert torch.allclose(conv(x, adj2.t()), out, atol=1e-6)

    if is_full_test():
        t = '(Tensor, Tensor) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert jit(x, edge_index).tolist() == out.tolist()

    if is_full_test() and torch_geometric.typing.WITH_TORCH_SPARSE:
        t = '(Tensor, SparseTensor) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert torch.allclose(jit(x, adj2.t()), out, atol=1e-6)


def test_eg_conv_multiple_aggregators():
    x = torch.randn(4, 16)
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])

    conv = EGConv(16, 32, aggregators=["max", "min"])
    assert str(conv) == "EGConv(16, 32, aggregators=['max', 'min'])"
    out = conv(x, edge_index)
    assert out.size() == (4, 32)

    if torch_geometric.typing.WITH_TORCH_SPARSE:
        adj = SparseTensor.from_edge_index(edge_index, sparse_sizes=(4, 4))
        assert torch.allclose(conv(x, adj.t()), out, atol=1e-6)


def test_eg_conv_with_sparse_input_feature():
    x = torch.randn(4, 16).to_sparse_coo()
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])

    conv = EGConv(16, 32)
    assert conv(x, edge_index).size() == (4, 32)
