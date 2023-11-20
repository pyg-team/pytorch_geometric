import torch

from torch_geometric.data import Data
from torch_geometric.transforms import RandomShear


def test_random_shear():
    assert str(RandomShear(0.1)) == 'RandomShear(0.1)'

    pos = torch.tensor([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])

    data = Data(pos=pos)
    data = RandomShear(0)(data)
    assert len(data) == 1
    assert torch.allclose(data.pos, pos)

    data = Data(pos=pos)
    data = RandomShear(0.1)(data)
    assert len(data) == 1
    assert not torch.allclose(data.pos, pos)
