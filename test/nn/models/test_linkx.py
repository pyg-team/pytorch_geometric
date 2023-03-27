import pytest
import torch

from torch_geometric.nn import LINKX
from torch_geometric.testing import is_full_test
from torch_geometric.typing import SparseTensor


@pytest.mark.parametrize('num_edge_layers', [1, 2])
def test_linkx(num_edge_layers):
    x = torch.randn(4, 16)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
    edge_weight = torch.rand(edge_index.size(1))
    adj2 = SparseTensor.from_edge_index(edge_index, edge_weight,
                                        sparse_sizes=(4, 4))
    adj1 = adj2.set_value(None)

    model = LINKX(num_nodes=4, in_channels=16, hidden_channels=32,
                  out_channels=8, num_layers=2,
                  num_edge_layers=num_edge_layers)
    assert str(model) == 'LINKX(num_nodes=4, in_channels=16, out_channels=8)'

    out = model(x, edge_index)
    assert out.size() == (4, 8)
    assert torch.allclose(out, model(x, adj1.t()), atol=1e-4)

    if is_full_test():
        t = '(OptTensor, Tensor, OptTensor) -> Tensor'
        jit = torch.jit.script(model.jittable(t))
        assert torch.allclose(jit(x, edge_index), out)

        t = '(OptTensor, SparseTensor, OptTensor) -> Tensor'
        jit = torch.jit.script(model.jittable(t))
        assert torch.allclose(jit(x, adj1.t()), out)

    out = model(None, edge_index)
    assert out.size() == (4, 8)
    assert torch.allclose(out, model(None, adj1.t()), atol=1e-4)

    out = model(x, edge_index, edge_weight)
    assert out.size() == (4, 8)
    assert torch.allclose(out, model(x, adj2.t()), atol=1e-4)

    out = model(None, edge_index, edge_weight)
    assert out.size() == (4, 8)
    assert torch.allclose(out, model(None, adj2.t()), atol=1e-4)
