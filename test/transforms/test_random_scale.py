import torch

from torch_geometric.data import Data
from torch_geometric.transforms import RandomScale


def test_random_scale():
    assert str(RandomScale([1, 2])) == 'RandomScale([1, 2])'

    pos = torch.tensor([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])

    data = Data(pos=pos)
    data = RandomScale([1, 1])(data)
    assert len(data) == 1
    assert data.pos.tolist() == pos.tolist()

    data = Data(pos=pos)
    data = RandomScale([2, 2])(data)
    assert len(data) == 1
    assert data.pos.tolist() == [[-2, -2], [-2, 2], [2, -2], [2, 2]]
