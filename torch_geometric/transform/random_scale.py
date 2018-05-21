import torch
from torch_geometric.transform import LinearTransformation


class RandomScale(object):
    r"""Scales node positions by a randomly sampled factor :math:`s` within a
    given interval, e.g., resulting in the transformation matrix

    .. math::
        \begin{bmatrix}
            s & 0 & 0 \\
            0 & s & 0 \\
            0 & 0 & s \\
        \end{bmatrix}

    for three-dimensional positions.

    Args:
        scale (tuple): scaling factor interval, e.g. :obj:`(a, b)`, then scale
            is randomly sampled from the range
            :math:`a \leq \mathrm{scale} \leq b`.
    """

    def __init__(self, scale):
        self.scale = scale

    def __call__(self, data):
        rand = data.pos.new_empty(1).uniform(*self.scale)
        matrix = torch.diag(rand.expand(data.pos.size(1)))

        return LinearTransformation(matrix)(data)

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, *self.scale)
