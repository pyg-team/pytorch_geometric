import torch
import torch.nn.functional as F

from torch_geometric.utils.cross_entropy import sparse_cross_entropy


def test_sparse_cross_entropy_multiclass():
    x = torch.randn(5, 5, requires_grad=True)
    y = torch.eye(5)
    edge_label_index = y.nonzero().t()

    expected = F.cross_entropy(x, y)
    expected.backward()
    expected_grad = x.grad

    x.grad = None
    out = sparse_cross_entropy(x, edge_label_index)
    out.backward()

    assert torch.allclose(expected, out)
    assert torch.allclose(expected_grad, x.grad)


def test_sparse_cross_entropy_multilabel():
    x = torch.randn(4, 4, requires_grad=True)
    y = torch.randint_like(x, 0, 2)
    edge_label_index = y.nonzero().t()

    expected = F.cross_entropy(x, y)
    expected.backward()
    expected_grad = x.grad

    x.grad = None
    out = sparse_cross_entropy(x, edge_label_index)
    out.backward()

    assert torch.allclose(expected, out)
    assert torch.allclose(expected_grad, x.grad)
