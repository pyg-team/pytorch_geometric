import torch
from torch_geometric.transforms import LinearTransformation
from torch_geometric.data import Data


def test_cartesian():
    matrix = torch.tensor([[2, 0], [0, 2]], dtype=torch.float)
    transform = LinearTransformation(matrix)
    assert transform.__repr__() == ('LinearTransformation('
                                    '[[2.0, 0.0], [0.0, 2.0]])')

    pos = torch.Tensor([[-1, 1], [-3, 0], [2, -1]])

    data = Data(pos=pos)
    data = transform(data)
    assert len(data) == 1
    assert data.pos.tolist() == [[-2, 2], [-6, 0], [4, -2]]
