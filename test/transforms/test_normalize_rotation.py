from math import sqrt

import torch

from torch_geometric.data import Data
from torch_geometric.transforms import NormalizeRotation


def test_normalize_rotation():
    assert str(NormalizeRotation()) == 'NormalizeRotation()'

    pos = torch.Tensor([[-2, -2], [-1, -1], [0, 0], [1, 1], [2, 2]])
    normal = torch.Tensor([[-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1]])
    data = Data(pos=pos)
    data.normal = normal
    data = NormalizeRotation()(data)
    assert len(data) == 2

    expected_pos = torch.Tensor([
        [-2 * sqrt(2), 0],
        [-sqrt(2), 0],
        [0, 0],
        [sqrt(2), 0],
        [2 * sqrt(2), 0],
    ])
    expected_normal = [[0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]

    assert torch.allclose(data.pos, expected_pos, atol=1e-04)
    assert data.normal.tolist() == expected_normal

    data = Data(pos=pos)
    data.normal = normal
    data = NormalizeRotation(max_points=3)(data)
    assert len(data) == 2

    assert torch.allclose(data.pos, expected_pos, atol=1e-04)
    assert data.normal.tolist() == expected_normal

    data = Data(pos=pos)
    data.normal = normal
    data = NormalizeRotation(sort=True)(data)
    assert len(data) == 2

    assert torch.allclose(data.pos, expected_pos, atol=1e-04)
    assert data.normal.tolist() == expected_normal
