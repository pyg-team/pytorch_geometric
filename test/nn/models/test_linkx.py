import pytest
import torch

import torch_geometric.typing
from torch_geometric.nn import LINKX
from torch_geometric.testing import is_full_test
from torch_geometric.typing import SparseTensor


@pytest.mark.parametrize('num_edge_layers', [1, 2])
def test_linkx(num_edge_layers):
    x = torch.randn(4, 16)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
    edge_weight = torch.rand(edge_index.size(1))

    model = LINKX(num_nodes=4, in_channels=16, hidden_channels=32,
                  out_channels=8, num_layers=2,
                  num_edge_layers=num_edge_layers)
    assert str(model) == 'LINKX(num_nodes=4, in_channels=16, out_channels=8)'

    out = model(x, edge_index)
    assert out.size() == (4, 8)
    if torch_geometric.typing.WITH_TORCH_SPARSE:
        adj = SparseTensor.from_edge_index(edge_index, sparse_sizes=(4, 4))
        assert torch.allclose(out, model(x, adj.t()), atol=1e-6)

    if is_full_test():
        jit = torch.jit.script(model.jittable())
        assert torch.allclose(jit(x, edge_index), out)

    if is_full_test() and torch_geometric.typing.WITH_TORCH_SPARSE:
        jit = torch.jit.script(model.jittable(use_sparse_tensor=True))
        assert torch.allclose(jit(x, adj.t()), out, atol=1e-6)

    out = model(None, edge_index)
    assert out.size() == (4, 8)
    if torch_geometric.typing.WITH_TORCH_SPARSE:
        assert torch.allclose(out, model(None, adj.t()), atol=1e-6)

    out = model(x, edge_index, edge_weight)
    assert out.size() == (4, 8)
    if torch_geometric.typing.WITH_TORCH_SPARSE:
        adj = SparseTensor.from_edge_index(edge_index, edge_weight,
                                           sparse_sizes=(4, 4))
        assert torch.allclose(model(x, adj.t()), out, atol=1e-6)

    out = model(None, edge_index, edge_weight)
    assert out.size() == (4, 8)
    if torch_geometric.typing.WITH_TORCH_SPARSE:
        assert torch.allclose(model(None, adj.t()), out, atol=1e-6)
