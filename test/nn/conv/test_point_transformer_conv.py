import torch
from torch.nn import Linear, ReLU, Sequential

import torch_geometric.typing
from torch_geometric.nn import PointTransformerConv
from torch_geometric.testing import is_full_test
from torch_geometric.typing import SparseTensor
from torch_geometric.utils import to_torch_csc_tensor


def test_point_transformer_conv():
    x1 = torch.rand(4, 16)
    x2 = torch.randn(2, 8)
    pos1 = torch.rand(4, 3)
    pos2 = torch.randn(2, 3)
    edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    adj1 = to_torch_csc_tensor(edge_index, size=(4, 4))

    conv = PointTransformerConv(in_channels=16, out_channels=32)
    assert str(conv) == 'PointTransformerConv(16, 32)'

    out = conv(x1, pos1, edge_index)
    assert out.size() == (4, 32)
    assert torch.allclose(conv(x1, pos1, adj1.t()), out, atol=1e-6)

    if torch_geometric.typing.WITH_TORCH_SPARSE:
        adj2 = SparseTensor.from_edge_index(edge_index, sparse_sizes=(4, 4))
        assert torch.allclose(conv(x1, pos1, adj2.t()), out, atol=1e-6)

    if is_full_test():
        t = '(Tensor, Tensor, Tensor) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert torch.allclose(jit(x1, pos1, edge_index), out, atol=1e-6)

    if is_full_test() and torch_geometric.typing.WITH_TORCH_SPARSE:
        t = '(Tensor, Tensor, SparseTensor) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert torch.allclose(jit(x1, pos1, adj2.t()), out, atol=1e-6)

    pos_nn = Sequential(Linear(3, 16), ReLU(), Linear(16, 32))
    attn_nn = Sequential(Linear(32, 32), ReLU(), Linear(32, 32))
    conv = PointTransformerConv(16, 32, pos_nn, attn_nn)

    out = conv(x1, pos1, edge_index)
    assert out.size() == (4, 32)
    assert torch.allclose(conv(x1, pos1, adj1.t()), out, atol=1e-6)
    if torch_geometric.typing.WITH_TORCH_SPARSE:
        assert torch.allclose(conv(x1, pos1, adj2.t()), out, atol=1e-6)

    # Test biparitite message passing:
    adj1 = to_torch_csc_tensor(edge_index, size=(4, 2))

    conv = PointTransformerConv((16, 8), 32)
    assert str(conv) == 'PointTransformerConv((16, 8), 32)'

    out = conv((x1, x2), (pos1, pos2), edge_index)
    assert out.size() == (2, 32)
    assert torch.allclose(conv((x1, x2), (pos1, pos2), adj1.t()), out)

    if torch_geometric.typing.WITH_TORCH_SPARSE:
        adj2 = SparseTensor.from_edge_index(edge_index, sparse_sizes=(4, 2))
        assert torch.allclose(conv((x1, x2), (pos1, pos2), adj2.t()), out)

    if is_full_test():
        t = '(PairTensor, PairTensor, Tensor) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert torch.allclose(jit((x1, x2), (pos1, pos2), edge_index), out)

    if is_full_test() and torch_geometric.typing.WITH_TORCH_SPARSE:
        t = '(PairTensor, PairTensor, SparseTensor) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert torch.allclose(jit((x1, x2), (pos1, pos2), adj2.t()), out)
