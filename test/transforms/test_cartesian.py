import torch
from torch_geometric.transforms import Cartesian
from torch_geometric.data import Data


def test_cartesian():
    assert Cartesian().__repr__() == 'Cartesian(cat=True)'

    pos = torch.Tensor([[-1, 0], [0, 0], [2, 0]])
    edge_index = torch.LongTensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    data = Data(edge_index=edge_index, pos=pos)
    expected_output = [[0.75, 0.5], [0.25, 0.5], [1, 0.5], [0, 0.5]]

    output = Cartesian()(data)
    assert output.edge_attr.tolist() == expected_output
