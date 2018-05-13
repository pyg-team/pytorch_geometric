import torch
from torch_geometric.transform import TargetIndegree
from torch_geometric.datasets.dataset import Data


def test_target_indegree():
    assert TargetIndegree().__repr__() == 'TargetIndegree(cat=True)'

    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    data = Data(None, None, edge_index, None, None)

    out = TargetIndegree()(data).weight.tolist()
    assert out == [[1], [0.5], [0.5], [1]]

    data.weight = torch.tensor([1, 1, 1, 1], dtype=torch.float)
    out = TargetIndegree()(data).weight.tolist()
    assert out == [[1, 1], [1, 0.5], [1, 0.5], [1, 1]]
