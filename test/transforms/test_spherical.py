from math import pi as PI

import torch

from torch_geometric.data import Data
from torch_geometric.transforms import Spherical


def test_spherical():
    assert str(Spherical()) == 'Spherical(norm=True, max_value=None)'

    pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    edge_index = torch.tensor([[0, 1], [1, 0]])
    edge_attr = torch.tensor([1.0, 1.0])

    data = Data(edge_index=edge_index, pos=pos)
    data = Spherical(norm=False)(data)
    assert len(data) == 3
    assert data.pos.tolist() == pos.tolist()
    assert data.edge_index.tolist() == edge_index.tolist()
    assert torch.allclose(
        data.edge_attr,
        torch.tensor([[1.0, 0.0, PI / 2.0], [1.0, PI, PI / 2.0]]),
        atol=1e-4,
    )

    data = Data(edge_index=edge_index, pos=pos, edge_attr=edge_attr)
    data = Spherical(norm=True)(data)
    assert len(data) == 3
    assert data.pos.tolist() == pos.tolist()
    assert data.edge_index.tolist() == edge_index.tolist()
    assert torch.allclose(
        data.edge_attr,
        torch.tensor([[1.0, 1.0, 0.0, 0.5], [1.0, 1.0, 0.5, 0.5]]),
        atol=1e-4,
    )

    pos = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    edge_index = torch.tensor([[0, 1], [1, 0]])

    data = Data(edge_index=edge_index, pos=pos)
    data = Spherical(norm=False)(data)
    assert len(data) == 3
    assert data.pos.tolist() == pos.tolist()
    assert data.edge_index.tolist() == edge_index.tolist()
    assert torch.allclose(
        data.edge_attr,
        torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, PI]]),
        atol=1e-4,
    )

    data = Data(edge_index=edge_index, pos=pos, edge_attr=edge_attr)
    data = Spherical(norm=True)(data)
    assert len(data) == 3
    assert data.pos.tolist() == pos.tolist()
    assert data.edge_index.tolist() == edge_index.tolist()
    assert torch.allclose(
        data.edge_attr,
        torch.tensor([[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 1.0]]),
        atol=1e-4,
    )
