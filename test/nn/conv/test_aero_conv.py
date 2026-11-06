import pytest
import torch

from torch_geometric.nn import AEROConv
from torch_geometric.testing import withPackage
from torch_geometric.typing import SparseTensor


@pytest.mark.parametrize('empty', [False, True])
def test_aero_conv_isolated_nodes(empty):
    x = torch.randn(5, 4, requires_grad=True)
    edge_index = torch.tensor([[0, 1], [1, 0]])
    if empty:
        edge_index = torch.empty(2, 0, dtype=torch.long)
    conv = AEROConv(4, 4, iterations=2)

    out = conv(x, edge_index)
    assert out.size() == (5, 4)
    assert torch.allclose(out, conv(x, edge_index, num_nodes=5))
    out.sum().backward()
    assert x.grad.isfinite().all()
    for param in conv.parameters():
        if param.grad is not None:
            assert param.grad.isfinite().all()


@withPackage('torch_sparse')
def test_aero_conv_sparse():
    x = torch.randn(5, 4)
    edge_index = torch.tensor([[0, 0, 1, 2], [1, 2, 2, 0]])
    adj = SparseTensor.from_edge_index(edge_index, sparse_sizes=(5, 5))
    conv = AEROConv(4, 2, heads=2, iterations=3)

    assert torch.allclose(conv(x, edge_index), conv(x, adj.t()), atol=1e-6)
