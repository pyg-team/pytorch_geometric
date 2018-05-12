from math import pi as PI

import torch


class Spherical(object):
    def __init__(self, cat=True):
        self.cat = cat

    def __call__(self, data):
        (row, col), pos, pseudo = data.index, data.pos, data.weight
        assert pos.dim() == 3

        cart = pos[col] - pos[row]
        rho = torch.norm(cart, p=2, dim=-1)
        rho /= rho.max()
        theta = torch.atan2(cart[..., 1], cart[..., 0]) / (2 * PI)
        theta += (theta < 0).type_as(theta)
        phi = torch.acos(cart[..., 2]) / PI
        polar = torch.stack([rho, theta, phi], dim=1)

        if pseudo is not None and self.cat:
            pseudo = pseudo.view(-1, 1) if pseudo.dim() == 1 else pseudo
            data.weight = torch.cat([pseudo, cart.type_as(polar)], dim=-1)
        else:
            data.weight = polar

        return data

    def __repr__(self):
        return '{}(cat={})'.format(self.__class__.__name__, self.cat)
