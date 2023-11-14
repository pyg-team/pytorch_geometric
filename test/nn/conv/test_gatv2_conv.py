import torch

import torch_geometric.typing
from torch_geometric.nn import GATv2Conv
from torch_geometric.testing import is_full_test
from torch_geometric.typing import SparseTensor
from torch_geometric.utils import to_torch_csc_tensor


def test_gatv2_conv():
    x1 = torch.randn(4, 8)
    x2 = torch.randn(2, 8)
    edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    adj1 = to_torch_csc_tensor(edge_index, size=(4, 4))

    conv = GATv2Conv(8, 32, heads=2)
    assert str(conv) == 'GATv2Conv(8, 32, heads=2)'
    out = conv(x1, edge_index)
    assert out.size() == (4, 64)
    assert torch.allclose(conv(x1, edge_index), out)
    assert torch.allclose(conv(x1, adj1.t()), out, atol=1e-6)

    if torch_geometric.typing.WITH_TORCH_SPARSE:
        adj2 = SparseTensor.from_edge_index(edge_index, sparse_sizes=(4, 4))
        assert torch.allclose(conv(x1, adj2.t()), out, atol=1e-6)

    if is_full_test():
        t = '(Tensor, Tensor, OptTensor, NoneType) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert torch.allclose(jit(x1, edge_index), out)

    if is_full_test() and torch_geometric.typing.WITH_TORCH_SPARSE:
        t = '(Tensor, SparseTensor, OptTensor, NoneType) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert torch.allclose(jit(x1, adj2.t()), out, atol=1e-6)

    # Test `return_attention_weights`.
    result = conv(x1, edge_index, return_attention_weights=True)
    assert torch.allclose(result[0], out)
    assert result[1][0].size() == (2, 7)
    assert result[1][1].size() == (7, 2)
    assert result[1][1].min() >= 0 and result[1][1].max() <= 1
    assert conv._alpha is None

    if torch_geometric.typing.WITH_PT113:
        # PyTorch < 1.13 does not support multi-dimensional CSR values :(
        result = conv(x1, adj1.t(), return_attention_weights=True)
        assert torch.allclose(result[0], out, atol=1e-6)
        assert result[1][0].size() == torch.Size([4, 4, 2])
        assert result[1][0]._nnz() == 7

    if torch_geometric.typing.WITH_TORCH_SPARSE:
        result = conv(x1, adj2.t(), return_attention_weights=True)
        assert torch.allclose(result[0], out, atol=1e-6)
        assert result[1].sizes() == [4, 4, 2] and result[1].nnz() == 7
        assert conv._alpha is None

    if is_full_test():
        t = ('(Tensor, Tensor, OptTensor, bool) -> '
             'Tuple[Tensor, Tuple[Tensor, Tensor]]')
        jit = torch.jit.script(conv.jittable(t))
        result = jit(x1, edge_index, return_attention_weights=True)
        assert torch.allclose(result[0], out)
        assert result[1][0].size() == (2, 7)
        assert result[1][1].size() == (7, 2)
        assert result[1][1].min() >= 0 and result[1][1].max() <= 1
        assert conv._alpha is None

    if is_full_test() and torch_geometric.typing.WITH_TORCH_SPARSE:
        t = ('(Tensor, SparseTensor, OptTensor, bool) -> '
             'Tuple[Tensor, SparseTensor]')
        jit = torch.jit.script(conv.jittable(t))
        result = jit(x1, adj2.t(), return_attention_weights=True)
        assert torch.allclose(result[0], out, atol=1e-6)
        assert result[1].sizes() == [4, 4, 2] and result[1].nnz() == 7
        assert conv._alpha is None

    # Test bipartite message passing:
    adj1 = to_torch_csc_tensor(edge_index, size=(4, 2))

    out = conv((x1, x2), edge_index)
    assert out.size() == (2, 64)
    assert torch.allclose(conv((x1, x2), edge_index), out)
    assert torch.allclose(conv((x1, x2), adj1.t()), out, atol=1e-6)

    if torch_geometric.typing.WITH_TORCH_SPARSE:
        adj2 = SparseTensor.from_edge_index(edge_index, sparse_sizes=(4, 2))
        assert torch.allclose(conv((x1, x2), adj2.t()), out, atol=1e-6)

    if is_full_test():
        t = '(OptPairTensor, Tensor, OptTensor, NoneType) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert torch.allclose(jit((x1, x2), edge_index), out)

    if is_full_test() and torch_geometric.typing.WITH_TORCH_SPARSE:
        t = '(OptPairTensor, SparseTensor, OptTensor, NoneType) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert torch.allclose(jit((x1, x2), adj2.t()), out, atol=1e-6)


def test_gatv2_conv_with_edge_attr():
    x = torch.randn(4, 8)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 1, 1]])
    edge_weight = torch.randn(edge_index.size(1))
    edge_attr = torch.randn(edge_index.size(1), 4)

    conv = GATv2Conv(8, 32, heads=2, edge_dim=1, fill_value=0.5)
    out = conv(x, edge_index, edge_weight)
    assert out.size() == (4, 64)

    conv = GATv2Conv(8, 32, heads=2, edge_dim=1, fill_value='mean')
    out = conv(x, edge_index, edge_weight)
    assert out.size() == (4, 64)

    conv = GATv2Conv(8, 32, heads=2, edge_dim=4, fill_value=0.5)
    out = conv(x, edge_index, edge_attr)
    assert out.size() == (4, 64)

    conv = GATv2Conv(8, 32, heads=2, edge_dim=4, fill_value='mean')
    out = conv(x, edge_index, edge_attr)
    assert out.size() == (4, 64)
