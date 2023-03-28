import pytest
import torch

import torch_geometric.typing
from torch_geometric.nn import DNAConv
from torch_geometric.testing import is_full_test
from torch_geometric.typing import SparseTensor
from torch_geometric.utils import to_torch_csc_tensor


@pytest.mark.parametrize('channels', [32])
@pytest.mark.parametrize('num_layers', [3])
def test_dna_conv(channels, num_layers):
    x = torch.randn((4, num_layers, channels))
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])

    conv = DNAConv(channels, heads=4, groups=8, dropout=0.0)
    assert str(conv) == 'DNAConv(32, heads=4, groups=8)'
    out = conv(x, edge_index)
    assert out.size() == (4, channels)

    if is_full_test():
        t = '(Tensor, Tensor, OptTensor) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert jit(x, edge_index).tolist() == out.tolist()

    conv = DNAConv(channels, heads=1, groups=1, dropout=0.0)
    assert str(conv) == 'DNAConv(32, heads=1, groups=1)'
    out = conv(x, edge_index)
    assert out.size() == (4, channels)

    if is_full_test():
        t = '(Tensor, Tensor, OptTensor) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert jit(x, edge_index).tolist() == out.tolist()

    conv = DNAConv(channels, heads=1, groups=1, dropout=0.0, cached=True)
    out = conv(x, edge_index)
    assert conv._cached_edge_index is not None
    out = conv(x, edge_index)
    assert out.size() == (4, channels)

    if is_full_test():
        t = '(Tensor, Tensor, OptTensor) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert jit(x, edge_index).tolist() == out.tolist()


@pytest.mark.parametrize('channels', [32])
@pytest.mark.parametrize('num_layers', [3])
def test_dna_conv_sparse_tensor(channels, num_layers):
    x = torch.randn((4, num_layers, channels))
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    value = torch.rand(edge_index.size(1))
    adj1 = to_torch_csc_tensor(edge_index, size=(4, 4))
    adj2 = to_torch_csc_tensor(edge_index, value, size=(4, 4))

    conv = DNAConv(32, heads=4, groups=8, dropout=0.0)
    assert str(conv) == 'DNAConv(32, heads=4, groups=8)'
    out1 = conv(x, edge_index)
    assert out1.size() == (4, 32)
    assert torch.allclose(conv(x, adj1.t()), out1, atol=1e-6)
    out2 = conv(x, edge_index, value)
    assert out2.size() == (4, 32)
    assert torch.allclose(conv(x, adj2.t()), out2, atol=1e-6)

    if torch_geometric.typing.WITH_TORCH_SPARSE:
        adj3 = SparseTensor.from_edge_index(edge_index, sparse_sizes=(4, 4))
        adj4 = SparseTensor.from_edge_index(edge_index, value, (4, 4))
        assert torch.allclose(conv(x, adj3.t()), out1, atol=1e-6)
        assert torch.allclose(conv(x, adj4.t()), out2, atol=1e-6)

    if is_full_test():
        t = '(Tensor, Tensor, OptTensor) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert jit(x, edge_index).tolist() == out1.tolist()
        assert jit(x, edge_index, value).tolist() == out2.tolist()

    if is_full_test() and torch_geometric.typing.WITH_TORCH_SPARSE:
        t = '(Tensor, SparseTensor, OptTensor) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert torch.allclose(jit(x, adj3.t()), out1, atol=1e-6)
        assert torch.allclose(jit(x, adj4.t()), out2, atol=1e-6)

    conv = DNAConv(channels, heads=1, groups=1, dropout=0.0, cached=True)

    out1 = conv(x, adj1.t())
    assert conv._cached_edge_index is not None
    assert torch.allclose(conv(x, adj1.t()), out1, atol=1e-6)

    if torch_geometric.typing.WITH_TORCH_SPARSE:
        assert torch.allclose(conv(x, adj3.t()), out1, atol=1e-6)
        assert conv._cached_adj_t is not None
        assert torch.allclose(conv(x, adj3.t()), out1, atol=1e-6)
