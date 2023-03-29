import torch
from torch.nn import Linear as Lin
from torch.nn import ReLU
from torch.nn import Sequential as Seq

import torch_geometric.typing
from torch_geometric.nn import PointNetConv
from torch_geometric.testing import is_full_test
from torch_geometric.typing import SparseTensor
from torch_geometric.utils import to_torch_csc_tensor


def test_point_net_conv():
    x1 = torch.randn(4, 16)
    pos1 = torch.randn(4, 3)
    pos2 = torch.randn(2, 3)
    edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    adj1 = to_torch_csc_tensor(edge_index, size=(4, 4))

    local_nn = Seq(Lin(16 + 3, 32), ReLU(), Lin(32, 32))
    global_nn = Seq(Lin(32, 32))
    conv = PointNetConv(local_nn, global_nn)
    assert str(conv) == (
        'PointNetConv(local_nn=Sequential(\n'
        '  (0): Linear(in_features=19, out_features=32, bias=True)\n'
        '  (1): ReLU()\n'
        '  (2): Linear(in_features=32, out_features=32, bias=True)\n'
        '), global_nn=Sequential(\n'
        '  (0): Linear(in_features=32, out_features=32, bias=True)\n'
        '))')
    out = conv(x1, pos1, edge_index)
    assert out.size() == (4, 32)
    assert torch.allclose(conv(x1, pos1, adj1.t()), out, atol=1e-6)

    if torch_geometric.typing.WITH_TORCH_SPARSE:
        adj2 = SparseTensor.from_edge_index(edge_index, sparse_sizes=(4, 4))
        assert torch.allclose(conv(x1, pos1, adj2.t()), out, atol=1e-6)

    if is_full_test():
        t = '(OptTensor, Tensor, Tensor) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert torch.allclose(jit(x1, pos1, edge_index), out, atol=1e-6)

    if is_full_test() and torch_geometric.typing.WITH_TORCH_SPARSE:
        t = '(OptTensor, Tensor, SparseTensor) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert torch.allclose(jit(x1, pos1, adj2.t()), out, atol=1e-6)

    # Test bipartite message passing:
    adj1 = to_torch_csc_tensor(edge_index, size=(4, 2))

    out = conv(x1, (pos1, pos2), edge_index)
    assert out.size() == (2, 32)
    assert torch.allclose(conv((x1, None), (pos1, pos2), edge_index), out)
    assert torch.allclose(conv(x1, (pos1, pos2), adj1.t()), out, atol=1e-6)
    assert torch.allclose(conv((x1, None), (pos1, pos2), adj1.t()), out)

    if torch_geometric.typing.WITH_TORCH_SPARSE:
        adj2 = SparseTensor.from_edge_index(edge_index, sparse_sizes=(4, 2))
        assert torch.allclose(conv(x1, (pos1, pos2), adj2.t()), out, atol=1e-6)
        assert torch.allclose(conv((x1, None), (pos1, pos2), adj2.t()), out)

    if is_full_test():
        t = '(PairOptTensor, PairTensor, Tensor) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert torch.allclose(jit(x1, (pos1, pos2), edge_index), out)

    if is_full_test() and torch_geometric.typing.WITH_TORCH_SPARSE:
        t = '(PairOptTensor, PairTensor, SparseTensor) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert torch.allclose(jit(x1, (pos1, pos2), adj2.t()), out, atol=1e-6)
