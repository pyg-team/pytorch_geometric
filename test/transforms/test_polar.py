from math import pi as PI

import torch

from torch_geometric.data import Data
from torch_geometric.transforms import Polar


def test_polar():
    assert str(Polar()) == 'Polar(norm=True, max_value=None)'

    pos = torch.Tensor([[0, 0], [1, 0]])
    edge_index = torch.tensor([[0, 1], [1, 0]])
    edge_attr = torch.Tensor([1, 1])

    data = Data(edge_index=edge_index, pos=pos)
    data = Polar(norm=False)(data)
    assert len(data) == 3
    assert data.pos.tolist() == pos.tolist()
    assert data.edge_index.tolist() == edge_index.tolist()
    assert torch.allclose(data.edge_attr, torch.Tensor([[1, 0], [1, PI]]),
                          atol=1e-04)

    data = Data(edge_index=edge_index, pos=pos, edge_attr=edge_attr)
    data = Polar(norm=True)(data)
    assert len(data) == 3
    assert data.pos.tolist() == pos.tolist()
    assert data.edge_index.tolist() == edge_index.tolist()
    assert torch.allclose(data.edge_attr,
                          torch.Tensor([[1, 1, 0], [1, 1, 0.5]]), atol=1e-04)
