import torch

from torch_geometric.data import Data
from torch_geometric.transforms import NormalizeScale


def test_normalize_scale():
    transform = NormalizeScale()
    assert str(transform) == 'NormalizeScale()'

    pos = torch.randn((10, 3))
    data = Data(pos=pos)

    data = transform(data)
    assert len(data) == 1
    assert data.pos.min().item() > -1
    assert data.pos.max().item() < 1


def test_normalize_scale_degenerate_positions():
    transform = NormalizeScale()
    pos = torch.full((4, 3), 7.0)

    data = transform(Data(pos=pos))

    assert torch.equal(data.pos, torch.zeros_like(pos))


def test_normalize_scale_empty_positions():
    transform = NormalizeScale()
    pos = torch.empty((0, 3))

    data = transform(Data(pos=pos))

    assert torch.equal(data.pos, pos)
