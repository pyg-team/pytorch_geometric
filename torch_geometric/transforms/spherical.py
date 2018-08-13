from math import pi as PI

import torch


class Spherical(object):
    r"""Saves the globally normalized three-dimensional spatial relation of
    linked nodes as spherical coordinates (mapped to the fixed interval
    :math:`[0, 1]`) in its edge attributes.

    Args:
        cat (bool, optional): Concat pseudo-coordinates to edge attributes
            instead of replacing them. (default: :obj:`True`)

    .. testsetup::

        import torch
        from torch_geometric.data import Data

    .. testcode::

        from torch_geometric.transforms import Spherical

        pos = torch.tensor([[0, 0, 0], [0, 1, 1]], dtype=torch.float)
        edge_index = torch.tensor([[0, 1], [1, 0]])
        data = Data(edge_index=edge_index, pos=pos)

        data = Spherical()(data)

        print(data.edge_attr)

    .. testoutput::

        tensor([[1.0000, 0.2500, 0.0000],
                [1.0000, 0.7500, 1.0000]])
    """

    def __init__(self, cat=True):
        self.cat = cat

    def __call__(self, data):
        (row, col), pos, pseudo = data.edge_index, data.pos, data.edge_attr
        assert pos.dim() == 2 and pos.size(1) == 3

        cart = pos[col] - pos[row]
        rho = torch.norm(cart, p=2, dim=-1)
        rho = rho / rho.max()
        theta = torch.atan2(cart[..., 1], cart[..., 0]) / (2 * PI)
        theta += (theta < 0).type_as(theta)
        phi = torch.acos(cart[..., 2] / rho) / PI
        spher = torch.stack([rho, theta, phi], dim=1)

        if pseudo is not None and self.cat:
            pseudo = pseudo.view(-1, 1) if pseudo.dim() == 1 else pseudo
            data.edge_attr = torch.cat([pseudo, spher.type_as(pos)], dim=-1)
        else:
            data.edge_attr = spher

        return data

    def __repr__(self):
        return '{}(cat={})'.format(self.__class__.__name__, self.cat)
