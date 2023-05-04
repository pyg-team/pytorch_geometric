import torch

import torch_geometric.typing
from torch_geometric.nn import BayesianGCNConv
from torch_geometric.testing import is_full_test
from torch_geometric.typing import SparseTensor
from torch_geometric.utils import to_torch_csc_tensor


def test_bayesian_gcn_conv():
    x = torch.randn(4, 16)
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    value = torch.rand(edge_index.size(1))
    adj1 = to_torch_csc_tensor(edge_index, size=(4, 4))
    adj2 = to_torch_csc_tensor(edge_index, value, size=(4, 4))

    conv = BayesianGCNConv(16, 32)
    assert str(conv) == "BayesianGCNConv(16, 32)"

    out1 = conv(x, edge_index)
    assert out1.size() == (4, 32)
    assert torch.allclose(
        conv(x, adj1.t()).mean().abs(),
        out1.mean().abs(), atol=1.0)
    assert torch.allclose(
        conv(x, adj1.t()).var().abs(),
        out1.var().abs(), atol=1.0)

    out2 = conv(x, edge_index, value)
    assert out2.size() == (4, 32)
    assert torch.allclose(
        conv(x, adj2.t()).mean().abs(),
        out2.mean().abs(), atol=1.0)
    assert torch.allclose(
        conv(x, adj2.t()).var().abs(),
        out2.var().abs(), atol=1.0)

    if torch_geometric.typing.WITH_TORCH_SPARSE:
        adj3 = SparseTensor.from_edge_index(edge_index, sparse_sizes=(4, 4))
        adj4 = SparseTensor.from_edge_index(edge_index, value, (4, 4))
        assert torch.allclose(
            conv(x, adj3.t()).mean().abs(),
            out1.mean().abs(), atol=1.0)
        assert torch.allclose(
            conv(x, adj3.t()).var().abs(),
            out1.var().abs(), atol=1.0)
        assert torch.allclose(
            conv(x, adj4.t()).mean().abs(),
            out2.mean().abs(), atol=1.0)
        assert torch.allclose(
            conv(x, adj4.t()).var().abs(),
            out2.var().abs(), atol=1.0)

    if is_full_test():
        t = "(Tensor, Tensor, OptTensor) -> Tensor"
        jit = torch.jit.script(conv.jittable(t))
        assert torch.allclose(
            jit(x, edge_index).mean().abs(),
            out1.mean().abs(), atol=1.0)
        assert torch.allclose(
            jit(x, edge_index).var().abs(),
            out1.var().abs(), atol=1.0)
        assert torch.allclose(
            jit(x, edge_index, value).mean().abs(),
            out2.mean().abs(), atol=1.0)
        assert torch.allclose(
            jit(x, edge_index, value).var().abs(),
            out2.var().abs(), atol=1.0)

    if is_full_test() and torch_geometric.typing.WITH_TORCH_SPARSE:
        t = "(Tensor, SparseTensor, OptTensor) -> Tensor"
        jit = torch.jit.script(conv.jittable(t))
        assert torch.allclose(
            jit(x, adj3.t()).mean().abs(),
            out1.mean().abs(), atol=1.0)
        assert torch.allclose(
            jit(x, adj3.t()).var().abs(),
            out1.var().abs(), atol=1.0)
        assert torch.allclose(
            jit(x, adj4.t()).mean().abs(),
            out2.mean().abs(), atol=1.0)
        assert torch.allclose(
            jit(x, adj4.t()).var().abs(),
            out2.var().abs(), atol=1.0)

    conv.cached = True
    conv(x, edge_index)
    assert conv._cached_edge_index is not None
    assert torch.allclose(
        conv(x, edge_index).mean().abs(),
        out1.mean().abs(), atol=1.0)
    assert torch.allclose(
        conv(x, adj1.t()).mean().abs(),
        out1.var().abs(), atol=1.0)
    assert torch.allclose(
        conv(x, adj1.t()).var().abs(),
        out1.mean().abs(), atol=1.0)
    assert torch.allclose(
        conv(x, edge_index).var().abs(),
        out1.var().abs(), atol=1.0)

    if torch_geometric.typing.WITH_TORCH_SPARSE:
        conv(x, adj3.t())
        assert conv._cached_adj_t is not None
        assert torch.allclose(
            conv(x, adj3.t()).mean().abs(),
            out1.mean().abs(), atol=1.0)
        assert torch.allclose(
            conv(x, adj3.t()).var().abs(),
            out1.var().abs(), atol=1.0)

    # Test `return_kl_divergence`.
    _, kl_div1 = conv(x, edge_index, return_kl_divergence=True)
    assert kl_div1.dim() == 0

    _, kl_div2 = conv(x, adj1.t(), return_kl_divergence=True)
    assert kl_div2.dim() == 0
    assert torch.allclose(kl_div1, kl_div2, atol=1e-6)

    if torch_geometric.typing.WITH_TORCH_SPARSE:
        _, kl_div3 = conv(x, adj2.t(), return_kl_divergence=True)
        assert kl_div3.dim() == 0
        assert torch.allclose(kl_div1, kl_div3, atol=1e-6)

    if is_full_test():
        t = ('(Tensor, Tensor, OptTensor, Size, bool) -> '
             'Tuple[Tensor, Tuple[Tensor, Tensor]]')
        jit = torch.jit.script(conv.jittable(t))
        _, kl_div4 = jit(x, edge_index, return_kl_divergence=True)
        assert kl_div3.dim() == 0
        assert torch.allclose(kl_div1, kl_div4, atol=1e-6)

    if is_full_test() and torch_geometric.typing.WITH_TORCH_SPARSE:
        t = ('(Tensor, SparseTensor, OptTensor, Size, bool) -> '
             'Tuple[Tensor, SparseTensor]')
        jit = torch.jit.script(conv.jittable(t))
        _, kl_div5 = jit(x, adj2.t(), return_kl_divergence=True)
        assert kl_div5.dim() == 0
        assert torch.allclose(kl_div1, kl_div5, atol=1e-6)
