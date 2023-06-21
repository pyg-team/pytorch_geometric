import pytest
import torch

import torch_geometric.typing
from torch_geometric.nn import HEATConv
from torch_geometric.testing import is_full_test
from torch_geometric.typing import SparseTensor


@pytest.mark.parametrize('concat', [True, False])
def test_heat_conv(concat):
    x = torch.randn(4, 8)
    edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    edge_attr = torch.randn((4, 2))
    node_type = torch.tensor([0, 0, 1, 2])
    edge_type = torch.tensor([0, 2, 1, 2])

    conv = HEATConv(in_channels=8, out_channels=16, num_node_types=3,
                    num_edge_types=3, edge_type_emb_dim=5, edge_dim=2,
                    edge_attr_emb_dim=6, heads=2, concat=concat)
    assert str(conv) == 'HEATConv(8, 16, heads=2)'

    out = conv(x, edge_index, node_type, edge_type, edge_attr)
    assert out.size() == (4, 32 if concat else 16)

    if torch_geometric.typing.WITH_TORCH_SPARSE:
        adj = SparseTensor.from_edge_index(edge_index, edge_attr, (4, 4))
        assert torch.allclose(conv(x, adj.t(), node_type, edge_type), out,
                              atol=1e-5)

    if is_full_test():
        t = '(Tensor, Tensor, Tensor, Tensor, OptTensor) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert torch.allclose(
            jit(x, edge_index, node_type, edge_type, edge_attr), out,
            atol=1e-5)

    if is_full_test() and torch_geometric.typing.WITH_TORCH_SPARSE:
        t = '(Tensor, SparseTensor, Tensor, Tensor, OptTensor) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert torch.allclose(jit(x, adj.t(), node_type, edge_type), out,
                              atol=1e-5)
