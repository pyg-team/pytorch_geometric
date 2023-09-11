import torch

import torch_geometric.typing
from torch_geometric.nn import WLConvContinuous
from torch_geometric.testing import is_full_test
from torch_geometric.typing import SparseTensor


def test_wl_conv():
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

    conv = WLConvContinuous()
    assert str(conv) == 'WLConvContinuous()'

    out = conv(x, edge_index)
    assert out.tolist() == [[-0.5], [0.0], [0.5]]

    if torch_geometric.typing.WITH_TORCH_SPARSE:
        adj = SparseTensor.from_edge_index(edge_index, sparse_sizes=(3, 3))
        assert torch.allclose(conv(x, adj.t()), out)

    if is_full_test():
        t = '(Tensor, Tensor, OptTensor, Size) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert torch.allclose(jit(x, edge_index), out)

    if is_full_test() and torch_geometric.typing.WITH_TORCH_SPARSE:
        t = '(Tensor, SparseTensor, OptTensor, Size) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert torch.allclose(jit(x, adj.t()), out, atol=1e-6)

    # Test bipartite message passing:
    x1 = torch.randn(4, 8)
    x2 = torch.randn(2, 8)
    edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    edge_weight = torch.randn(edge_index.size(1))

    out1 = conv((x1, None), edge_index, edge_weight, size=(4, 2))
    assert out1.size() == (2, 8)

    out2 = conv((x1, x2), edge_index, edge_weight)
    assert out2.size() == (2, 8)

    if torch_geometric.typing.WITH_TORCH_SPARSE:
        adj = SparseTensor.from_edge_index(edge_index, edge_weight, (4, 2))
        assert torch.allclose(conv((x1, None), adj.t()), out1)
        assert torch.allclose(conv((x1, x2), adj.t()), out2)

    if is_full_test():
        t = '(OptPairTensor, Tensor, OptTensor, Size) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert torch.allclose(
            jit((x1, None), edge_index, edge_weight, size=(4, 2)), out1)
        assert torch.allclose(jit((x1, x2), edge_index, edge_weight), out2)

    if is_full_test() and torch_geometric.typing.WITH_TORCH_SPARSE:
        t = '(OptPairTensor, SparseTensor, OptTensor, Size) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert torch.allclose(jit((x1, None), adj.t()), out1, atol=1e-6)
        assert torch.allclose(jit((x1, x2), adj.t()), out2, atol=1e-6)
