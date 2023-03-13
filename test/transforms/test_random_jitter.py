import torch

from torch_geometric.data import Data
from torch_geometric.transforms import RandomJitter


def test_random_jitter():
    assert str(RandomJitter(0.1)) == 'RandomJitter(0.1)'

    pos = torch.Tensor([[0, 0], [0, 0], [0, 0], [0, 0]])

    data = Data(pos=pos)
    data = RandomJitter(0)(data)
    assert len(data) == 1
    assert data.pos.tolist() == pos.tolist()

    data = Data(pos=pos)
    data = RandomJitter(0.1)(data)
    assert len(data) == 1
    assert data.pos.min().item() >= -0.1
    assert data.pos.max().item() <= 0.1

    data = Data(pos=pos)
    data = RandomJitter([0.1, 1])(data)
    assert len(data) == 1
    assert data.pos[:, 0].min().item() >= -0.1
    assert data.pos[:, 0].max().item() <= 0.1
    assert data.pos[:, 1].min().item() >= -1
    assert data.pos[:, 1].max().item() <= 1
