import torch

from torch_geometric.data import Data
from torch_geometric.transforms import RandomRotate


def test_random_rotate():
    assert str(RandomRotate([-180, 180])) == ('RandomRotate('
                                              '[-180, 180], axis=0)')

    pos = torch.Tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]])

    data = Data(pos=pos)
    data = RandomRotate(0)(data)
    assert len(data) == 1
    assert data.pos.tolist() == pos.tolist()

    data = Data(pos=pos)
    data = RandomRotate([180, 180])(data)
    assert len(data) == 1
    assert data.pos.tolist() == [[1, 1], [1, -1], [-1, 1], [-1, -1]]

    pos = torch.Tensor([[-1, -1, 1], [-1, 1, 1], [1, -1, -1], [1, 1, -1]])

    data = Data(pos=pos)
    data = RandomRotate([180, 180], axis=0)(data)
    assert len(data) == 1
    assert data.pos.tolist() == [[-1, 1, -1], [-1, -1, -1], [1, 1, 1],
                                 [1, -1, 1]]

    data = Data(pos=pos)
    data = RandomRotate([180, 180], axis=1)(data)
    assert len(data) == 1
    assert data.pos.tolist() == [[1, -1, -1], [1, 1, -1], [-1, -1, 1],
                                 [-1, 1, 1]]

    data = Data(pos=pos)
    data = RandomRotate([180, 180], axis=2)(data)
    assert len(data) == 1
    assert data.pos.tolist() == [[1, 1, 1], [1, -1, 1], [-1, 1, -1],
                                 [-1, -1, -1]]
