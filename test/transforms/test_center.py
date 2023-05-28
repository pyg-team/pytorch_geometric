import torch

from torch_geometric.data import Data
from torch_geometric.transforms import Center


def test_center():
    transform = Center()
    assert str(transform) == 'Center()'

    pos = torch.Tensor([[0, 0], [2, 0], [4, 0]])
    data = Data(pos=pos)

    data = transform(data)
    assert len(data) == 1
    assert data.pos.tolist() == [[-2, 0], [0, 0], [2, 0]]
