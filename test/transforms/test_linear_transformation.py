import pytest
import torch

from torch_geometric.data import Data
from torch_geometric.transforms import LinearTransformation


@pytest.mark.parametrize('matrix', [
    [[2.0, 0.0], [0.0, 2.0]],
    torch.tensor([[2.0, 0.0], [0.0, 2.0]]),
])
def test_linear_transformation(matrix):
    pos = torch.tensor([[-1.0, 1.0], [-3.0, 0.0], [2.0, -1.0]])

    transform = LinearTransformation(matrix)
    assert str(transform) == ('LinearTransformation(\n'
                              '[[2. 0.]\n'
                              ' [0. 2.]]\n'
                              ')')

    out = transform(Data(pos=pos))
    assert len(out) == 1
    assert torch.allclose(out.pos, 2 * pos)

    out = transform(Data())
    assert len(out) == 0
