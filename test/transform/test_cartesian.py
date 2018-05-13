import torch
from torch_geometric.transform import Cartesian
from torch_geometric.datasets.dataset import Data


def test_cartesian():
    assert Cartesian().__repr__() == 'Cartesian(cat=True)'

    pos = torch.tensor([[-1, 0], [0, 0], [2, 0]], dtype=torch.float)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    data = Data(None, pos, edge_index, None, None)

    out = Cartesian()(data).weight.tolist()
    assert out == [[0.75, 0.5], [0.25, 0.5], [1, 0.5], [0, 0.5]]

    data.weight = torch.tensor([1, 1, 1, 1], dtype=torch.float)
    out = Cartesian()(data).weight.tolist()
    assert out == [[1, 0.75, 0.5], [1, 0.25, 0.5], [1, 1, 0.5], [1, 0, 0.5]]
