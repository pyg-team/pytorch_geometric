import pytest
import torch
import torch.nn.functional as F

from torch_geometric.utils.cross_entropy import sparse_cross_entropy


@pytest.mark.parametrize('with_edge_label_weight', [False, True])
def test_sparse_cross_entropy_multiclass(with_edge_label_weight):
    x = torch.randn(5, 5, requires_grad=True)
    y = torch.eye(5)

    edge_label_index = y.nonzero().t()
    edge_label_weight = None
    if with_edge_label_weight:
        edge_label_weight = torch.rand(edge_label_index.size(1))
        y[y == 1.0] = edge_label_weight

    expected = F.cross_entropy(x, y)
    expected.backward()
    expected_grad = x.grad

    x.grad = None
    out = sparse_cross_entropy(x, edge_label_index, edge_label_weight)
    out.backward()

    assert torch.allclose(expected, out)
    assert torch.allclose(expected_grad, x.grad)


@pytest.mark.parametrize('with_edge_label_weight', [False, True])
def test_sparse_cross_entropy_multilabel(with_edge_label_weight):
    x = torch.randn(4, 4, requires_grad=True)
    y = torch.randint_like(x, 0, 2)

    edge_label_index = y.nonzero().t()
    edge_label_weight = None
    if with_edge_label_weight:
        edge_label_weight = torch.rand(edge_label_index.size(1))
        y[y == 1.0] = edge_label_weight

    expected = F.cross_entropy(x, y)
    expected.backward()
    expected_grad = x.grad

    x.grad = None
    out = sparse_cross_entropy(x, edge_label_index, edge_label_weight)
    out.backward()

    assert torch.allclose(expected, out)
    assert torch.allclose(expected_grad, x.grad)
