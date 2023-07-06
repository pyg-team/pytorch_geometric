from math import sqrt

import pytest
import torch

from torch_geometric.testing import withPackage
from torch_geometric.utils import geodesic_distance


@withPackage('gdist')
@pytest.mark.skip(reason="No way of currently testing this")
def test_geodesic_distance():
    pos = torch.tensor([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [2.0, 2.0, 0.0],
    ])
    face = torch.tensor([[0, 1, 3], [0, 2, 3]]).t()

    out = geodesic_distance(pos, face)
    expected = torch.tensor([
        [0.0, 1.0, 1.0, sqrt(2)],
        [1.0, 0.0, sqrt(2), 1.0],
        [1.0, sqrt(2), 0.0, 1.0],
        [sqrt(2), 1.0, 1.0, 0.0],
    ])
    assert torch.allclose(out, expected)
    assert torch.allclose(out, geodesic_distance(pos, face, num_workers=-1))

    out = geodesic_distance(pos, face, norm=False)
    expected = [
        [0, 2, 2, 2 * sqrt(2)],
        [2, 0, 2 * sqrt(2), 2],
        [2, 2 * sqrt(2), 0, 2],
        [2 * sqrt(2), 2, 2, 0],
    ]
    assert torch.allclose(out, expected)

    src = torch.tensor([0, 0, 0, 0])
    dest = torch.tensor([0, 1, 2, 3])
    out = geodesic_distance(pos, face, src=src, dest=dest)
    expected = torch.tensor([0.0, 1.0, 1.0, sqrt(2)])
    assert torch.allclose(out, expected)

    out = geodesic_distance(pos, face, dest=dest)
    expected = torch.tensor([0.0, 0.0, 0.0, 0.0])
    assert torch.allclose(out, expected)
