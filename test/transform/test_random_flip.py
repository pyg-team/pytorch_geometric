import torch
from torch_geometric.transform import RandomFlip
from torch_geometric.datasets.dataset import Data


def test_cartesian():
    assert RandomFlip(axis=0).__repr__() == 'RandomFlip(axis=0, p=0.5)'

    pos = torch.tensor([[-1, 1], [-3, 0], [2, -1]], dtype=torch.float)
    data = Data(None, pos, None, None, None)
    expected_output = [[1, 1], [3, 0], [-2, -1]]

    data = RandomFlip(axis=0, p=1)(data)
    assert data.pos.tolist() == expected_output
