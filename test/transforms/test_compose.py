import torch
import torch_geometric.transforms as T
from torch_geometric.data import Data


def test_compose():
    transform = T.Compose([T.Cartesian(), T.TargetIndegree()])
    assert transform.__repr__() == ('Compose([\n'
                                    '    Cartesian(cat=True),\n'
                                    '    TargetIndegree(cat=True),\n'
                                    '])')

    pos = torch.tensor([[-1, 0], [0, 0], [2, 0]], dtype=torch.float)
    edge_index = torch.tensor([[0, 1], [1, 2]])
    data = Data(edge_index=edge_index, pos=pos)

    out = transform(data).edge_attr.tolist()
    assert out == [[0.75, 0.5, 1], [1, 0.5, 1]]
