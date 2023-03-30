from copy import copy

import torch

from torch_geometric.data import Data
from torch_geometric.transforms import FixedPoints


def test_fixed_points():
    assert str(FixedPoints(1024)) == 'FixedPoints(1024, replace=True)'

    data = Data(
        pos=torch.randn(100, 3),
        x=torch.randn(100, 16),
        y=torch.randn(1),
        edge_attr=torch.randn(100, 3),
        num_nodes=100,
    )

    out = FixedPoints(50, replace=True)(copy(data))
    assert len(out) == 5
    assert out.pos.size() == (50, 3)
    assert out.x.size() == (50, 16)
    assert out.y.size() == (1, )
    assert out.edge_attr.size() == (100, 3)
    assert out.num_nodes == 50

    out = FixedPoints(200, replace=True)(copy(data))
    assert len(out) == 5
    assert out.pos.size() == (200, 3)
    assert out.x.size() == (200, 16)
    assert out.y.size() == (1, )
    assert out.edge_attr.size() == (100, 3)
    assert out.num_nodes == 200

    out = FixedPoints(50, replace=False, allow_duplicates=False)(copy(data))
    assert len(out) == 5
    assert out.pos.size() == (50, 3)
    assert out.x.size() == (50, 16)
    assert out.y.size() == (1, )
    assert out.edge_attr.size() == (100, 3)
    assert out.num_nodes == 50

    out = FixedPoints(200, replace=False, allow_duplicates=False)(copy(data))
    assert len(out) == 5
    assert out.pos.size() == (100, 3)
    assert out.x.size() == (100, 16)
    assert out.y.size() == (1, )
    assert out.edge_attr.size() == (100, 3)
    assert out.num_nodes == 100

    out = FixedPoints(50, replace=False, allow_duplicates=True)(copy(data))
    assert len(out) == 5
    assert out.pos.size() == (50, 3)
    assert out.x.size() == (50, 16)
    assert out.y.size() == (1, )
    assert out.edge_attr.size() == (100, 3)
    assert out.num_nodes == 50

    out = FixedPoints(200, replace=False, allow_duplicates=True)(copy(data))
    assert len(out) == 5
    assert out.pos.size() == (200, 3)
    assert out.x.size() == (200, 16)
    assert out.y.size() == (1, )
    assert out.edge_attr.size() == (100, 3)
    assert out.num_nodes == 200
