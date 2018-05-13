from math import pi as PI

import torch


class Polar(object):
    r"""Saves the globally normalized two-dimensional spatial relation of
    linked nodes as polar coordinates (mapped to the fixed interval
    :math:`[0, 1]`) in its edge attributes.

    Args:
        cat (bool, optional): Concat pseudo-coordinates to edge attributes
            instead of replacing them. (default: :obj:`True`)

    .. testsetup::

        import torch
        from torch_geometric.datasets.dataset import Data

    .. testcode::

        from torch_geometric.transform import Polar

        pos = torch.tensor([[-1, 0], [0, 0], [0, 2]], dtype=torch.float)
        edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
        data = Data(None, pos, edge_index, None, None)

        data = Polar()(data)

        print(data.weight)

    .. testoutput::

        tensor([[ 0.5000,  0.0000],
                [ 0.5000,  0.5000],
                [ 1.0000,  0.2500],
                [ 1.0000,  0.7500]])
    """

    def __init__(self, cat=True):
        self.cat = cat

    def __call__(self, data):
        (row, col), pos, pseudo = data.index, data.pos, data.weight
        assert pos.dim() == 2 and pos.size(1) == 2

        cart = pos[col] - pos[row]
        rho = torch.norm(cart, p=2, dim=-1)
        rho /= rho.max()
        theta = torch.atan2(cart[..., 1], cart[..., 0]) / (2 * PI)
        theta += (theta < 0).type_as(theta)
        polar = torch.stack([rho, theta], dim=1)

        if pseudo is not None and self.cat:
            pseudo = pseudo.view(-1, 1) if pseudo.dim() == 1 else pseudo
            data.weight = torch.cat([pseudo, polar.type_as(pos)], dim=-1)
        else:
            data.weight = polar

        return data

    def __repr__(self):
        return '{}(cat={})'.format(self.__class__.__name__, self.cat)
