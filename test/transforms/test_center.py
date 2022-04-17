import torch

from torch_geometric.data import Data
from torch_geometric.transforms import Center


def test_center():
    assert Center().__repr__() == 'Center()'

    pos = torch.Tensor([[0, 0], [2, 0], [4, 0]])

    data = Data(pos=pos)
    data = Center()(data)
    assert len(data) == 1
    assert data.pos.tolist() == [[-2, 0], [0, 0], [2, 0]]
