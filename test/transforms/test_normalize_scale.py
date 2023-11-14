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
