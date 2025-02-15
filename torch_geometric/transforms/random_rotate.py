import math
import random
from typing import Tuple, Union

import torch

from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform, LinearTransformation


@functional_transform('random_rotate')
class RandomRotate(BaseTransform):
    r"""Rotates node positions around a specific axis by a randomly sampled
    factor within a given interval (functional name: :obj:`random_rotate`).

    Args:
        degrees (tuple or float): Rotation interval from which the rotation
            angle is sampled. If :obj:`degrees` is a number instead of a
            tuple, the interval is given by :math:`[-\mathrm{degrees},
            \mathrm{degrees}]`.
        axis (int, optional): The rotation axis. (default: :obj:`0`)
    """
    def __init__(
        self,
        degrees: Union[Tuple[float, float], float],
        axis: int = 0,
    ) -> None:
        if isinstance(degrees, (int, float)):
            degrees = (-abs(degrees), abs(degrees))
        assert isinstance(degrees, (tuple, list)) and len(degrees) == 2
        self.degrees = degrees
        self.axis = axis

    def forward(self, data: Data) -> Data:
        assert data.pos is not None

        degree = math.pi * random.uniform(*self.degrees) / 180.0
        sin, cos = math.sin(degree), math.cos(degree)

        if data.pos.size(-1) == 2:
            matrix = [[cos, sin], [-sin, cos]]
        else:
            if self.axis == 0:
                matrix = [[1, 0, 0], [0, cos, sin], [0, -sin, cos]]
            elif self.axis == 1:
                matrix = [[cos, 0, -sin], [0, 1, 0], [sin, 0, cos]]
            else:
                matrix = [[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]]

        return LinearTransformation(torch.tensor(matrix))(data)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.degrees}, '
                f'axis={self.axis})')
