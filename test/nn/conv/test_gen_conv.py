import pytest
import torch

import torch_geometric.typing
from torch_geometric.nn import GENConv
from torch_geometric.testing import is_full_test
from torch_geometric.typing import SparseTensor
from torch_geometric.utils import to_torch_coo_tensor


@pytest.mark.parametrize('aggr', [
    'softmax',
    'powermean',
    ['softmax', 'powermean'],
])
def test_gen_conv(aggr):
    x1 = torch.randn(4, 16)
    x2 = torch.randn(2, 16)
    edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    value = torch.randn(edge_index.size(1), 16)
    adj1 = to_torch_coo_tensor(edge_index, size=(4, 4))
    adj2 = to_torch_coo_tensor(edge_index, value, size=(4, 4))

    conv = GENConv(16, 32, aggr, edge_dim=16, msg_norm=True)
    assert str(conv) == f'GENConv(16, 32, aggr={aggr})'
    out1 = conv(x1, edge_index)
    assert out1.size() == (4, 32)
    assert torch.allclose(conv(x1, edge_index, size=(4, 4)), out1)
    assert torch.allclose(conv(x1, adj1.t().coalesce()), out1)

    out2 = conv(x1, edge_index, value)
    assert out2.size() == (4, 32)
    assert torch.allclose(conv(x1, edge_index, value, (4, 4)), out2)
    # t() expects a tensor with <= 2 sparse and 0 dense dimensions
    assert torch.allclose(conv(x1, adj2.transpose(1, 0).coalesce()), out2)

    if torch_geometric.typing.WITH_TORCH_SPARSE:
        adj3 = SparseTensor.from_edge_index(edge_index, sparse_sizes=(4, 4))
        adj4 = SparseTensor.from_edge_index(edge_index, value, (4, 4))
        assert torch.allclose(conv(x1, adj3.t()), out1)
        assert torch.allclose(conv(x1, adj4.t()), out2)

    if is_full_test():
        t = '(Tensor, Tensor, OptTensor, Size) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert torch.allclose(jit(x1, edge_index), out1, atol=1e-6)
        assert torch.allclose(jit(x1, edge_index, size=(4, 4)), out1,
                              atol=1e-6)
        assert torch.allclose(jit(x1, edge_index, value), out2, atol=1e-6)
        assert torch.allclose(jit(x1, edge_index, value, size=(4, 4)), out2,
                              atol=1e-6)

    if is_full_test() and torch_geometric.typing.WITH_TORCH_SPARSE:
        t = '(Tensor, SparseTensor, OptTensor, Size) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert torch.allclose(jit(x1, adj3.t()), out1)
        assert torch.allclose(jit(x1, adj4.t()), out2)

    # Test bipartite message passing:
    adj1 = to_torch_coo_tensor(edge_index, size=(4, 2))
    adj2 = to_torch_coo_tensor(edge_index, value, size=(4, 2))

    out1 = conv((x1, x2), edge_index)
    assert out1.size() == (2, 32)
    assert torch.allclose(conv((x1, x2), edge_index, size=(4, 2)), out1)
    assert torch.allclose(conv((x1, x2), adj1.t().coalesce()), out1)

    out2 = conv((x1, x2), edge_index, value)
    assert out2.size() == (2, 32)
    assert torch.allclose(conv((x1, x2), edge_index, value, (4, 2)), out2)
    assert torch.allclose(conv((x1, x2),
                               adj2.transpose(1, 0).coalesce()), out2)

    if torch_geometric.typing.WITH_TORCH_SPARSE:
        adj3 = SparseTensor.from_edge_index(edge_index, sparse_sizes=(4, 2))
        adj4 = SparseTensor.from_edge_index(edge_index, value, (4, 2))
        assert torch.allclose(conv((x1, x2), adj3.t()), out1)
        assert torch.allclose(conv((x1, x2), adj4.t()), out2)

    if is_full_test():
        t = '(OptPairTensor, Tensor, OptTensor, Size) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert torch.allclose(jit((x1, x2), edge_index), out1, atol=1e-6)
        assert torch.allclose(jit((x1, x2), edge_index, size=(4, 2)), out1,
                              atol=1e-6)
        assert torch.allclose(jit((x1, x2), edge_index, value), out2,
                              atol=1e-6)
        assert torch.allclose(jit((x1, x2), edge_index, value, (4, 2)), out2,
                              atol=1e-6)

    if is_full_test() and torch_geometric.typing.WITH_TORCH_SPARSE:
        t = '(OptPairTensor, SparseTensor, OptTensor, Size) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert torch.allclose(jit((x1, x2), adj3.t()), out1)
        assert torch.allclose(jit((x1, x2), adj4.t()), out2)

    # Test bipartite message passing with unequal feature dimensions:
    conv.reset_parameters()
    assert float(conv.msg_norm.scale) == 1

    x1 = torch.randn(4, 8)
    x2 = torch.randn(2, 16)

    conv = GENConv((8, 16), 32, aggr)
    assert str(conv) == f'GENConv((8, 16), 32, aggr={aggr})'

    out1 = conv((x1, x2), edge_index)
    assert out1.size() == (2, 32)
    assert torch.allclose(conv((x1, x2), edge_index, size=(4, 2)), out1)
    assert torch.allclose(conv((x1, x2), adj1.t().coalesce()), out1)

    out2 = conv((x1, None), edge_index, size=(4, 2))
    assert out2.size() == (2, 32)
    assert torch.allclose(conv((x1, None), adj1.t().coalesce()), out2)

    if torch_geometric.typing.WITH_TORCH_SPARSE:
        assert torch.allclose(conv((x1, x2), adj3.t()), out1)
        assert torch.allclose(conv((x1, None), adj3.t()), out2)

    # Test lazy initialization:
    conv = GENConv((-1, -1), 32, aggr, edge_dim=-1)
    assert str(conv) == f'GENConv((-1, -1), 32, aggr={aggr})'
    out1 = conv((x1, x2), edge_index, value)
    assert out1.size() == (2, 32)
    assert torch.allclose(conv((x1, x2), edge_index, value, size=(4, 2)), out1)
    assert torch.allclose(conv((x1, x2),
                               adj2.transpose(1, 0).coalesce()), out1)

    out2 = conv((x1, None), edge_index, value, size=(4, 2))
    assert out2.size() == (2, 32)
    assert torch.allclose(conv((x1, None),
                               adj2.transpose(1, 0).coalesce()), out2)

    if torch_geometric.typing.WITH_TORCH_SPARSE:
        assert torch.allclose(conv((x1, x2), adj4.t()), out1)
        assert torch.allclose(conv((x1, None), adj4.t()), out2)

    if is_full_test():
        t = '(OptPairTensor, Tensor, OptTensor, Size) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert torch.allclose(jit((x1, x2), edge_index, value), out1,
                              atol=1e-6)
        assert torch.allclose(jit((x1, x2), edge_index, value, size=(4, 2)),
                              out1, atol=1e-6)
        assert torch.allclose(jit((x1, None), edge_index, value, size=(4, 2)),
                              out2, atol=1e-6)

    if is_full_test() and torch_geometric.typing.WITH_TORCH_SPARSE:
        t = '(OptPairTensor, SparseTensor, OptTensor, Size) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert torch.allclose(jit((x1, x2), adj4.t()), out1, atol=1e-6)
        assert torch.allclose(jit((x1, None), adj4.t()), out2, atol=1e-6)
