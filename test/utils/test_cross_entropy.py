import torch
import torch.nn.functional as F

from torch_geometric.utils.cross_entropy import sparse_cross_entropy


def test_sparse_cross_entropy_multiclass():
    x = torch.randn(5, 5)
    y = torch.eye(5)
    edge_label_index = y.nonzero().t()

    expected = F.cross_entropy(x, y)
    out = sparse_cross_entropy(x, edge_label_index)
    assert torch.allclose(expected, out)


def test_sparse_cross_entropy_multilabel():
    x = torch.randn(5, 8)
    y = torch.randint_like(x, 0, 2)
    edge_label_index = y.nonzero().t()

    expected = F.cross_entropy(x, y)
    out = sparse_cross_entropy(x, edge_label_index)
    assert torch.allclose(expected, out)
