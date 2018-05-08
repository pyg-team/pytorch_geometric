import torch
from torch_geometric.transform import LocalCartesian
from torch_geometric.datasets.dataset import Data


def test_local_cartesian():
    assert LocalCartesian().__repr__() == 'LocalCartesian(cat=True)'

    pos = torch.Tensor([[-1, 0], [0, 0], [2, 0]])
    edge_index = torch.LongTensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    data = Data(None, pos, edge_index, None, None)
    expected_output = [[1, 0.5], [0.25, 0.5], [1, 0.5], [0, 0.5]]

    data = LocalCartesian()(data)
    assert data.weight.tolist() == expected_output
