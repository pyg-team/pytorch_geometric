import random
from typing import Tuple

from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform


@functional_transform('random_scale')
class RandomScale(BaseTransform):
    r"""Scales node positions by a randomly sampled factor :math:`s` within a
    given interval, *e.g.*, resulting in the transformation matrix
    (functional name: :obj:`random_scale`)

    .. math::
        \begin{bmatrix}
            s & 0 & 0 \\
            0 & s & 0 \\
            0 & 0 & s \\
        \end{bmatrix}

    for three-dimensional positions.

    Args:
        scales (tuple): scaling factor interval, e.g. :obj:`(a, b)`, then scale
            is randomly sampled from the range
            :math:`a \leq \mathrm{scale} \leq b`.
    """
    def __init__(self, scales: Tuple[float, float]):
        assert isinstance(scales, (tuple, list)) and len(scales) == 2
        self.scales = scales

    def forward(self, data: Data) -> Data:
        scale = random.uniform(*self.scales)
        data.pos = data.pos * scale
        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.scales})'
