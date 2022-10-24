from typing import Union

import torch

from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform, LinearTransformation


@functional_transform('random_shear')
class RandomShear(BaseTransform):
    r"""Shears node positions by randomly sampled factors :math:`s` within a
    given interval, *e.g.*, resulting in the transformation matrix
    (functional name: :obj:`random_shear`)

    .. math::
        \begin{bmatrix}
            1      & s_{xy} & s_{xz} \\
            s_{yx} & 1      & s_{yz} \\
            s_{zx} & z_{zy} & 1      \\
        \end{bmatrix}

    for three-dimensional positions.

    Args:
        shear (float or int): maximum shearing factor defining the range
            :math:`(-\mathrm{shear}, +\mathrm{shear})` to sample from.
    """
    def __init__(self, shear: Union[float, int]):
        self.shear = abs(shear)

    def __call__(self, data: Data) -> Data:
        dim = data.pos.size(-1)

        matrix = data.pos.new_empty(dim, dim).uniform_(-self.shear, self.shear)
        eye = torch.arange(dim, dtype=torch.long)
        matrix[eye, eye] = 1

        return LinearTransformation(matrix)(data)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.shear})'
