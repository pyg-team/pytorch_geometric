import torch
import torch_geometric.transforms as T
from torch_geometric.data import Data


def test_compose():
    transform = T.Compose([T.Cartesian(cat=False), T.Cartesian(cat=True)])
    expected_output = ('Compose([\n'
                       '    Cartesian(cat=False),\n'
                       '    Cartesian(cat=True),\n'
                       '])')
    assert transform.__repr__() == expected_output

    pos = torch.Tensor([[-1, 0], [0, 0], [2, 0]])
    edge_index = torch.LongTensor([[0, 1], [1, 2]])
    data = Data(edge_index=edge_index, pos=pos)
    expected_output = [[0.75, 0.5, 0.75, 0.5], [1, 0.5, 1, 0.5]]

    output = transform(data)
    assert output.edge_attr.tolist() == expected_output
