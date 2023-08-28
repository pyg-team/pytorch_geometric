import torch

import torch_geometric.typing
from torch_geometric.nn import MFConv
from torch_geometric.testing import is_full_test
from torch_geometric.typing import SparseTensor


def test_mf_conv():
    x1 = torch.randn(4, 8)
    x2 = torch.randn(2, 16)
    edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])

    conv = MFConv(8, 32)
    assert str(conv) == 'MFConv(8, 32)'
    out = conv(x1, edge_index)
    assert out.size() == (4, 32)
    assert torch.allclose(conv(x1, edge_index, size=(4, 4)), out)

    if torch_geometric.typing.WITH_TORCH_SPARSE:
        adj = SparseTensor.from_edge_index(edge_index, sparse_sizes=(4, 4))
        assert torch.allclose(conv(x1, adj.t()), out)

    if is_full_test():
        t = '(Tensor, Tensor, Size) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert torch.allclose(jit(x1, edge_index), out)
        assert torch.allclose(jit(x1, edge_index, size=(4, 4)), out)

    if is_full_test() and torch_geometric.typing.WITH_TORCH_SPARSE:
        t = '(Tensor, SparseTensor, Size) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert torch.allclose(jit(x1, adj.t()), out)

    # Test bipartite message passing:
    conv = MFConv((8, 16), 32)
    assert str(conv) == 'MFConv((8, 16), 32)'

    out1 = conv((x1, x2), edge_index)
    assert out1.size() == (2, 32)
    assert torch.allclose(conv((x1, x2), edge_index, (4, 2)), out1)

    out2 = conv((x1, None), edge_index, (4, 2))
    assert out2.size() == (2, 32)

    if torch_geometric.typing.WITH_TORCH_SPARSE:
        adj = SparseTensor.from_edge_index(edge_index, sparse_sizes=(4, 2))
        assert torch.allclose(conv((x1, x2), adj.t()), out1)
        assert torch.allclose(conv((x1, None), adj.t()), out2)

    if is_full_test():
        t = '(OptPairTensor, Tensor, Size) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert torch.allclose(jit((x1, x2), edge_index), out1)
        assert torch.allclose(jit((x1, x2), edge_index, size=(4, 2)), out1)
        assert torch.allclose(jit((x1, None), edge_index, size=(4, 2)), out2)

    if is_full_test() and torch_geometric.typing.WITH_TORCH_SPARSE:
        t = '(OptPairTensor, SparseTensor, Size) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert torch.allclose(jit((x1, x2), adj.t()), out1)
        assert torch.allclose(jit((x1, None), adj.t()), out2)
