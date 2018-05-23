import torch
from torch_geometric.transforms import LinearTransformation
from torch_geometric.data import Data


def test_cartesian():
    matrix = torch.tensor([[2, 0], [0, 2]], dtype=torch.float)
    transform = LinearTransformation(matrix)
    out = 'LinearTransformation([[2.0, 0.0], [0.0, 2.0]])'
    assert transform.__repr__() == out

    pos = torch.tensor([[-1, 1], [-3, 0], [2, -1]], dtype=torch.float)
    data = Data(pos=pos)

    out = transform(data).pos.tolist()
    assert out == [[-2, 2], [-6, 0], [4, -2]]
