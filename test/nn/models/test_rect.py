import torch

from torch_geometric.nn import RECT_L


def test_rect():
    x = torch.randn(6, 8)
    y = torch.tensor([1, 0, 0, 2, 1, 1])
    edge_index = torch.tensor([[0, 1, 1, 2, 4, 5], [1, 0, 2, 1, 5, 4]])
    mask = torch.randint(0, 2, (6, ), dtype=torch.bool)

    model = RECT_L(8, 16)
    assert str(model) == 'RECT_L(8, 16)'

    out = model(x, edge_index)
    assert out.size() == (6, 8)

    # Test `embed`:
    out = model.embed(x, edge_index)
    assert out.size() == (6, 16)

    # Test `get_semantic_labels`:
    out = model.get_semantic_labels(x, y, mask)
    assert out.size() == (int(mask.sum()), 8)
