import pytest
import torch

from torch_geometric.data import Data
from torch_geometric.transforms import LinearTransformation


@pytest.mark.parametrize('matrix', [
    [[2., 0.], [0., 2.]],
    torch.tensor([[2., 0.], [0., 2.]]),
])
def test_linear_transformation(matrix):
    pos = torch.Tensor([[-1, 1], [-3, 0], [2, -1]])

    transform = LinearTransformation(matrix)
    assert str(transform) == ('LinearTransformation(\n'
                              '[[2. 0.]\n'
                              ' [0. 2.]]\n'
                              ')')

    out = transform(Data(pos=pos))
    assert len(out) == 1
    assert out.pos.tolist() == [[-2, 2], [-6, 0], [4, -2]]

    out = transform(Data())
    assert len(out) == 0
