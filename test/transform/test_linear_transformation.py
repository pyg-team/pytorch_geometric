import torch
from torch_geometric.transform import LinearTransformation
from torch_geometric.datasets.dataset import Data


def test_cartesian():
    matrix = torch.tensor([[2, 0], [0, 2]], dtype=torch.float)
    transform = LinearTransformation(matrix)
    assert transform.__repr__() == ('LinearTransformation([\n'
                                    '    [2. 0.]\n'
                                    '    [0. 2.]\n'
                                    '])')

    pos = torch.tensor([[-1, 1], [-3, 0], [2, -1]], dtype=torch.float)
    data = Data(None, pos, None, None, None)

    out = transform(data).pos.tolist()
    assert out == [[-2, 2], [-6, 0], [4, -2]]
