import pytest
import torch

from torch_geometric.nn.models import RECT_L

dropouts = [0.0, 0.5]


@pytest.mark.parametrize('dropout', dropouts)
def test_rect(dropout):
    x = torch.randn(6, 8)
    y = torch.tensor([1, 0, 0, 2, 1, 1])
    edge_index = torch.tensor([[0, 1, 1, 2, 4, 5], [1, 0, 2, 1, 5, 4]])
    mask = torch.randint(0, 2, (6, ), dtype=torch.bool)

    model = RECT_L(8, 16, dropout=dropout)
    assert str(model) == 'RECT_L(8, 16)'

    out = model(x, edge_index)
    assert out.size() == (6, 8)

    # Test embed
    out = model.embed(x, edge_index)
    assert out.size() == (6, 16)
    assert torch.allclose(model.embed(x, edge_index), out)

    # Test get_semantic_labels
    out = model.get_semantic_labels(x, y, mask)
    assert out.size() == (mask.sum(), 8)
    assert torch.allclose(model.get_semantic_labels(x, y, mask), out)
