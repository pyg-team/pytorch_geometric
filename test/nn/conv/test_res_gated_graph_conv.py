import pytest
import torch

import torch_geometric.typing
from torch_geometric.nn import ResGatedGraphConv
from torch_geometric.testing import is_full_test
from torch_geometric.typing import SparseTensor
from torch_geometric.utils import to_torch_csc_tensor


@pytest.mark.parametrize('edge_dim', [None, 4])
def test_res_gated_graph_conv(edge_dim):
    x1 = torch.randn(4, 8)
    x2 = torch.randn(2, 32)
    edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    edge_attr = torch.randn(edge_index.size(1), edge_dim) if edge_dim else None
    adj1 = to_torch_csc_tensor(edge_index, size=(4, 4))

    conv = ResGatedGraphConv(8, 32, edge_dim=edge_dim)
    assert str(conv) == 'ResGatedGraphConv(8, 32)'

    out = conv(x1, edge_index, edge_attr)
    assert out.size() == (4, 32)
    assert torch.allclose(conv(x1, adj1.t(), edge_attr), out, atol=1e-6)

    if torch_geometric.typing.WITH_TORCH_SPARSE:
        adj2 = SparseTensor.from_edge_index(edge_index, edge_attr, (4, 4))
        assert torch.allclose(conv(x1, adj2.t()), out, atol=1e-6)

    if is_full_test():
        t = '(Tensor, Tensor, OptTensor) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert torch.allclose(jit(x1, edge_index, edge_attr), out, atol=1e-6)

    if is_full_test() and torch_geometric.typing.WITH_TORCH_SPARSE:
        t = '(Tensor, SparseTensor, OptTensor) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert torch.allclose(jit(x1, adj2.t()), out, atol=1e-6)

    # Test bipartite message passing:
    adj1 = to_torch_csc_tensor(edge_index, size=(4, 2))

    conv = ResGatedGraphConv((8, 32), 32, edge_dim=edge_dim)
    assert str(conv) == 'ResGatedGraphConv((8, 32), 32)'

    out = conv((x1, x2), edge_index, edge_attr)
    assert out.size() == (2, 32)
    assert torch.allclose(conv((x1, x2), adj1.t(), edge_attr), out, atol=1e-6)

    if torch_geometric.typing.WITH_TORCH_SPARSE:
        adj2 = SparseTensor.from_edge_index(edge_index, edge_attr, (4, 2))
        assert torch.allclose(conv((x1, x2), adj2.t()), out, atol=1e-6)

    if is_full_test():
        t = '(PairTensor, Tensor, OptTensor) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert torch.allclose(jit((x1, x2), edge_index, edge_attr), out,
                              atol=1e-6)

    if is_full_test() and torch_geometric.typing.WITH_TORCH_SPARSE:
        t = '(PairTensor, SparseTensor, OptTensor) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert torch.allclose(jit((x1, x2), adj2.t()), out, atol=1e-6)
