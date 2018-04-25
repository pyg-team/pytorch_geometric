import torch
from torch_geometric.transforms import TargetIndegree
from torch_geometric.data import Data


def test_cartesian():
    assert TargetIndegree().__repr__() == 'TargetIndegree(cat=True)'

    edge_index = torch.LongTensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    data = Data(edge_index=edge_index)
    expected_output = [1, 0.5, 0.5, 1]

    output = TargetIndegree()(data)
    assert output.edge_attr.tolist() == expected_output
