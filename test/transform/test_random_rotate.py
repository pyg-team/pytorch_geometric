from pytest import approx
import torch
from torch_geometric.transform import RandomRotate
from torch_geometric.datasets.dataset import Data


def test_spherical():
    assert RandomRotate(-180).__repr__() == 'RandomRotate((-180, 180))'

    pos = torch.tensor([[1, 0], [0, 1]], dtype=torch.float)
    data = Data(None, pos, None, None, None)

    out = RandomRotate((90, 90))(data).pos.view(-1).tolist()
    assert approx(out) == [0, 1, -1, 0]
