import torch

import torch_geometric.typing
from torch_geometric import seed_everything
from torch_geometric.nn import FiLMConv, FastFiLMConv
from torch_geometric.testing import is_full_test
from torch_geometric.typing import SparseTensor


def test_film_conv():
    seed_everything(42)
    x1 = torch.randn(4, 4)
    x2 = torch.randn(2, 16)
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3], [0, 0, 1, 0, 1, 1]])
    edge_type = torch.tensor([0, 1, 1, 0, 0, 1])
    conv = FiLMConv(4, 32)
    assert str(conv) == 'FiLMConv(4, 32, num_relations=1)'
    out = conv(x1, edge_index)
    assert out.size() == (4, 32)

    if torch_geometric.typing.WITH_TORCH_SPARSE:
        adj = SparseTensor.from_edge_index(edge_index, sparse_sizes=(4, 4))
        assert torch.allclose(conv(x1, adj.t()), out, atol=1e-6)

    if is_full_test():
        t = '(Tensor, Tensor, OptTensor) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert torch.allclose(jit(x1, edge_index), out, atol=1e-6)

    if is_full_test() and torch_geometric.typing.WITH_TORCH_SPARSE:
        t = '(Tensor, SparseTensor, OptTensor) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert torch.allclose(jit(x1, adj.t()), out, atol=1e-6)
    seed_everything(42)
    conv = FiLMConv(4, 32, num_relations=2)
    seed_everything(42)
    fast_conv = FastFiLMConv(4, 32, num_relations=2)
    assert str(conv) == 'FiLMConv(4, 32, num_relations=2)'
    assert str(fast_conv) == 'FastFiLMConv(4, 32, num_relations=2)'
    out = conv(x1, edge_index, edge_type)
    fast_out = fast_conv(x1, edge_index, edge_type)
    assert out.size() == (4, 32)
    assert fast_out.size() == (4, 32)
    for reg_param, fast_param in zip(conv.parameters(), fast_conv.parameters()):
        reg_param.data = torch.ones_like(reg_param.data)
        fast_param.data = torch.ones_like(fast_param.data)
    out = conv(x1, edge_index, edge_type)
    fast_out = fast_conv(x1, edge_index, edge_type)
    assert torch.allclose(out, fast_out), "max abs diff = " + str((out - fast_out).abs().max())
    if torch_geometric.typing.WITH_TORCH_SPARSE:
        adj = SparseTensor.from_edge_index(edge_index, edge_type, (4, 4))
        assert torch.allclose(conv(x1, adj.t()), out, atol=1e-6)

    if is_full_test():
        t = '(Tensor, Tensor, OptTensor) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert torch.allclose(jit(x1, edge_index, edge_type), out, atol=1e-6)

    if is_full_test() and torch_geometric.typing.WITH_TORCH_SPARSE:
        t = '(Tensor, SparseTensor, OptTensor) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert torch.allclose(jit(x1, adj.t()), out, atol=1e-6)

    # Test bipartite message passing:
    conv = FiLMConv((4, 16), 32)
    assert str(conv) == 'FiLMConv((4, 16), 32, num_relations=1)'
    out = conv((x1, x2), edge_index)
    assert out.size() == (2, 32)

    if torch_geometric.typing.WITH_TORCH_SPARSE:
        adj = SparseTensor.from_edge_index(edge_index, sparse_sizes=(4, 2))
        assert torch.allclose(conv((x1, x2), adj.t()), out, atol=1e-6)

    if is_full_test():
        t = '(PairTensor, Tensor, OptTensor) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert torch.allclose(jit((x1, x2), edge_index), out, atol=1e-6)

    if is_full_test() and torch_geometric.typing.WITH_TORCH_SPARSE:
        t = '(PairTensor, SparseTensor, OptTensor) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert torch.allclose(jit((x1, x2), adj.t()), out, atol=1e-6)

    conv = FiLMConv((4, 16), 32, num_relations=2)
    assert str(conv) == 'FiLMConv((4, 16), 32, num_relations=2)'
    out = conv((x1, x2), edge_index, edge_type)
    assert out.size() == (2, 32)

    if torch_geometric.typing.WITH_TORCH_SPARSE:
        adj = SparseTensor.from_edge_index(edge_index, edge_type, (4, 2))
        assert torch.allclose(conv((x1, x2), adj.t()), out, atol=1e-6)

    if is_full_test():
        t = '(PairTensor, Tensor, OptTensor) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert torch.allclose(jit((x1, x2), edge_index, edge_type), out,
                              atol=1e-6)

    if is_full_test() and torch_geometric.typing.WITH_TORCH_SPARSE:
        t = '(PairTensor, SparseTensor, OptTensor) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert torch.allclose(jit((x1, x2), adj.t()), out, atol=1e-6)

