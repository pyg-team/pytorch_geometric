import pytest
import torch
from torch_geometric.transform import Polar
from torch_geometric.datasets.dataset import Data


def test_polar():
    assert Polar().__repr__() == 'Polar(cat=True)'

    pos = torch.tensor([[-1, 0], [0, 0], [0, 2]], dtype=torch.float)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    data = Data(None, pos, edge_index, None, None)
    expected_output = [0.5, 0, 0.5, 0.5, 1, 0.25, 1, 0.75]

    data = Polar()(data)
    assert pytest.approx(data.weight.view(-1).tolist()) == expected_output
