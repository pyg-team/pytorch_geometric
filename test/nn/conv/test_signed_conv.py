import torch

import torch_geometric.typing
from torch_geometric.nn import SignedConv
from torch_geometric.testing import is_full_test
from torch_geometric.typing import SparseTensor
from torch_geometric.utils import to_torch_csc_tensor


def test_signed_conv():
    x = torch.randn(4, 16)
    edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    adj1 = to_torch_csc_tensor(edge_index, size=(4, 4))

    conv1 = SignedConv(16, 32, first_aggr=True)
    assert str(conv1) == 'SignedConv(16, 32, first_aggr=True)'

    conv2 = SignedConv(32, 48, first_aggr=False)
    assert str(conv2) == 'SignedConv(32, 48, first_aggr=False)'

    out1 = conv1(x, edge_index, edge_index)
    assert out1.size() == (4, 64)
    assert torch.allclose(conv1(x, adj1.t(), adj1.t()), out1)

    if torch_geometric.typing.WITH_TORCH_SPARSE:
        adj2 = SparseTensor.from_edge_index(edge_index, sparse_sizes=(4, 4))
        assert torch.allclose(conv1(x, adj2.t(), adj2.t()), out1)

    out2 = conv2(out1, edge_index, edge_index)
    assert out2.size() == (4, 96)
    assert torch.allclose(conv2(out1, adj1.t(), adj1.t()), out2)

    if torch_geometric.typing.WITH_TORCH_SPARSE:
        assert torch.allclose(conv2(out1, adj2.t(), adj2.t()), out2)

    if is_full_test():
        t = '(Tensor, Tensor, Tensor) -> Tensor'
        jit1 = torch.jit.script(conv1.jittable(t))
        jit2 = torch.jit.script(conv2.jittable(t))
        assert torch.allclose(jit1(x, edge_index, edge_index), out1)
        assert torch.allclose(jit2(out1, edge_index, edge_index), out2)

    if is_full_test() and torch_geometric.typing.WITH_TORCH_SPARSE:
        t = '(Tensor, SparseTensor, SparseTensor) -> Tensor'
        jit1 = torch.jit.script(conv1.jittable(t))
        jit2 = torch.jit.script(conv2.jittable(t))
        assert torch.allclose(jit1(x, adj2.t(), adj1.t()), out1)
        assert torch.allclose(jit2(out1, adj2.t(), adj1.t()), out2)

    # Test bipartite message passing:
    adj1 = to_torch_csc_tensor(edge_index, size=(4, 2))

    assert torch.allclose(conv1((x, x[:2]), edge_index, edge_index), out1[:2])
    assert torch.allclose(conv1((x, x[:2]), adj1.t(), adj1.t()), out1[:2])
    assert torch.allclose(conv2((out1, out1[:2]), edge_index, edge_index),
                          out2[:2], atol=1e-6)
    assert torch.allclose(conv2((out1, out1[:2]), adj1.t(), adj1.t()),
                          out2[:2], atol=1e-6)

    if torch_geometric.typing.WITH_TORCH_SPARSE:
        adj2 = SparseTensor.from_edge_index(edge_index, sparse_sizes=(4, 2))
        assert torch.allclose(conv1((x, x[:2]), adj2.t(), adj2.t()), out1[:2])
        assert torch.allclose(conv2((out1, out1[:2]), adj2.t(), adj2.t()),
                              out2[:2], atol=1e-6)

    if is_full_test():
        t = '(PairTensor, Tensor, Tensor) -> Tensor'
        jit1 = torch.jit.script(conv1.jittable(t))
        jit2 = torch.jit.script(conv2.jittable(t))
        assert torch.allclose(jit1((x, x[:2]), edge_index, edge_index),
                              out1[:2], atol=1e-6)
        assert torch.allclose(jit2((out1, out1[:2]), edge_index, edge_index),
                              out2[:2], atol=1e-6)

    if is_full_test() and torch_geometric.typing.WITH_TORCH_SPARSE:
        t = '(PairTensor, SparseTensor, SparseTensor) -> Tensor'
        jit1 = torch.jit.script(conv1.jittable(t))
        jit2 = torch.jit.script(conv2.jittable(t))
        assert torch.allclose(jit1((x, x[:2]), adj2.t(), adj1.t()), out1[:2])
        assert torch.allclose(jit2((out1, out1[:2]), adj2.t(), adj1.t()),
                              out2[:2], atol=1e-6)
