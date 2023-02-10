import torch

from torch_geometric.data import Data
from torch_geometric.transforms import LinearTransformation


def test_linear_transformation():
    pos = torch.Tensor([[-1, 1], [-3, 0], [2, -1]])
    data = Data(pos=pos)
    
    # expected __repr__ output
    expected = ('LinearTransformation(\n'
                '[[2. 0.]\n'
                ' [0. 2.]]\n'
                ')')

    # Test with matrix as 2D array:
    matrix = [[2., 0.], [0., 2.]]
    transform = LinearTransformation(matrix)
    assert str(transform) == expected

    # Test with matrix as Tensor:
    matrix = torch.tensor([[2, 0], [0, 2]], dtype=torch.float)
    transform = LinearTransformation(matrix)
    assert str(transform) == expected
    data = transform(data)
    assert len(data) == 1
    assert data.pos.tolist() == [[-2, 2], [-6, 0], [4, -2]]

    # Test without pos:
    data = Data()
    data = transform(data)
    assert len(data) == 0
