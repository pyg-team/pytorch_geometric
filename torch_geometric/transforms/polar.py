from math import pi as PI

import torch
from torch_geometric.transforms import Distance


class Polar(object):
    r"""Saves the relative polar coordinates of linked nodes in its edge
    attributes.

    Args:
        norm (bool, optional): If set to :obj:`False`, the output will not be
            normalized to lay in :math:`{[0, 1]}^2`. (default: :obj:`True`)
        max_value (float, optional): If set and :obj:`norm=True`, normalization
            will be performed based on this value instead of the maximum value
            found in the data. (default: :obj:`None`)
        cat (bool, optional): If set to :obj:`False`, all existing edge
            attributes will be replaced. (default: :obj:`True`)

    .. testsetup::

        import torch
        from torch_geometric.data import Data

    .. testcode::

        from torch_geometric.transforms import Polar

        pos = torch.Tensor([[-1, 0], [0, 0], [0, 2]])
        edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
        data = Data(edge_index=edge_index, pos=pos)

        data = Polar()(data)
    """

    def __init__(self, norm=True, max_value=None, cat=True):
        self.norm = norm
        self.max_value = max_value
        self.cat = cat
        self.distance = Distance(norm, max_value, cat=True)

    def __call__(self, data):
        (row, col), pos, pseudo = data.edge_index, data.pos, data.edge_attr
        assert pos.dim() == 2 and pos.size(1) == 2

        cart = pos[col] - pos[row]

        rho = torch.norm(cart, p=2, dim=-1).view(-1, 1)

        theta = torch.atan2(cart[..., 1], cart[..., 0]).view(-1, 1)
        theta = theta + (theta < 0).type_as(theta) + (2 * PI)

        if self.norm:
            if self.max_value is None:
                max_value = rho.max().item()
            else:
                max_value = self.max_value
            rho = rho / max_value
            theta = theta / (2 * PI)

        polar = torch.cat([rho, theta], dim=-1)

        if pseudo is not None and self.cat:
            pseudo = pseudo.view(-1, 1) if pseudo.dim() == 1 else pseudo
            data.edge_attr = torch.cat([pseudo, polar.type_as(pos)], dim=-1)
        else:
            data.edge_attr = polar

        return data

    def __repr__(self):
        return '{}(norm={}, max_value={})'.format(self.__class__.__name__,
                                                  self.norm, self.max_value)
