import torch
from torch.nn import Linear as Lin
from torch.nn import ReLU
from torch.nn import Sequential as Seq

import torch_geometric.typing
from torch_geometric.nn import DynamicEdgeConv, EdgeConv
from torch_geometric.testing import is_full_test, withPackage
from torch_geometric.typing import SparseTensor
from torch_geometric.utils import to_torch_csc_tensor


def test_edge_conv_conv():
    x1 = torch.randn(4, 16)
    x2 = torch.randn(2, 16)
    edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    adj1 = to_torch_csc_tensor(edge_index, size=(4, 4))

    nn = Seq(Lin(32, 16), ReLU(), Lin(16, 32))
    conv = EdgeConv(nn)
    assert str(conv) == (
        'EdgeConv(nn=Sequential(\n'
        '  (0): Linear(in_features=32, out_features=16, bias=True)\n'
        '  (1): ReLU()\n'
        '  (2): Linear(in_features=16, out_features=32, bias=True)\n'
        '))')
    out = conv(x1, edge_index)
    assert out.size() == (4, 32)
    assert torch.allclose(conv((x1, x1), edge_index), out, atol=1e-6)
    assert torch.allclose(conv(x1, adj1.t()), out, atol=1e-6)
    assert torch.allclose(conv((x1, x1), adj1.t()), out, atol=1e-6)

    if torch_geometric.typing.WITH_TORCH_SPARSE:
        adj2 = SparseTensor.from_edge_index(edge_index, sparse_sizes=(4, 4))
        assert torch.allclose(conv(x1, adj2.t()), out, atol=1e-6)
        assert torch.allclose(conv((x1, x1), adj2.t()), out, atol=1e-6)

    if is_full_test():
        t = '(Tensor, Tensor) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert torch.allclose(jit(x1, edge_index), out, atol=1e-6)

        t = '(PairTensor, Tensor) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert torch.allclose(jit((x1, x1), edge_index), out, atol=1e-6)

    if is_full_test() and torch_geometric.typing.WITH_TORCH_SPARSE:
        t = '(Tensor, SparseTensor) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert torch.allclose(jit(x1, adj2.t()), out, atol=1e-6)

        t = '(PairTensor, SparseTensor) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert torch.allclose(jit((x1, x1), adj2.t()), out, atol=1e-6)

    # Test bipartite message passing:
    adj1 = to_torch_csc_tensor(edge_index, size=(4, 2))

    out = conv((x1, x2), edge_index)
    assert out.size() == (2, 32)
    assert torch.allclose(conv((x1, x2), adj1.t()), out, atol=1e-6)

    if torch_geometric.typing.WITH_TORCH_SPARSE:
        adj2 = SparseTensor.from_edge_index(edge_index, sparse_sizes=(4, 2))
        assert torch.allclose(conv((x1, x2), adj2.t()), out, atol=1e-6)

    if is_full_test():
        t = '(PairTensor, Tensor) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert torch.allclose(jit((x1, x2), edge_index), out, atol=1e-6)

    if is_full_test() and torch_geometric.typing.WITH_TORCH_SPARSE:
        t = '(PairTensor, SparseTensor) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert torch.allclose(jit((x1, x2), adj2.t()), out, atol=1e-6)


@withPackage('torch_cluster')
def test_dynamic_edge_conv_conv():
    x1 = torch.randn(8, 16)
    x2 = torch.randn(4, 16)
    batch1 = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    batch2 = torch.tensor([0, 0, 1, 1])

    nn = Seq(Lin(32, 16), ReLU(), Lin(16, 32))
    conv = DynamicEdgeConv(nn, k=2)
    assert str(conv) == (
        'DynamicEdgeConv(nn=Sequential(\n'
        '  (0): Linear(in_features=32, out_features=16, bias=True)\n'
        '  (1): ReLU()\n'
        '  (2): Linear(in_features=16, out_features=32, bias=True)\n'
        '), k=2)')
    out11 = conv(x1)
    assert out11.size() == (8, 32)

    out12 = conv(x1, batch1)
    assert out12.size() == (8, 32)

    out21 = conv((x1, x2))
    assert out21.size() == (4, 32)

    out22 = conv((x1, x2), (batch1, batch2))
    assert out22.size() == (4, 32)

    if is_full_test():
        t = '(Tensor, OptTensor) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert torch.allclose(jit(x1), out11)
        assert torch.allclose(jit(x1, batch1), out12)

        t = '(PairTensor, Optional[PairTensor]) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert torch.allclose(jit((x1, x2)), out21)
        assert torch.allclose(jit((x1, x2), (batch1, batch2)), out22)

        torch.jit.script(conv.jittable())  # Test without explicit typing.
