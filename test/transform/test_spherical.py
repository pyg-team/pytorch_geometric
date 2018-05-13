from pytest import approx
import torch
from torch_geometric.transform import Spherical
from torch_geometric.datasets.dataset import Data


def test_spherical():
    assert Spherical().__repr__() == 'Spherical(cat=True)'

    pos = torch.tensor([[0, 0, 0], [0, 1, 1]], dtype=torch.float)
    edge_index = torch.tensor([[0, 1], [1, 0]])
    data = Data(None, pos, edge_index, None, None)

    out = Spherical()(data).weight.view(-1).tolist()
    assert approx(out) == [1, 0.25, 0, 1, 0.75, 1]

    data.weight = torch.tensor([1, 1], dtype=torch.float)
    out = Spherical()(data).weight.view(-1).tolist()
    assert approx(out) == [1, 1, 0.25, 0, 1, 1, 0.75, 1]
