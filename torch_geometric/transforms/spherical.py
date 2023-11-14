from math import pi as PI
from typing import Optional

import torch

from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform


@functional_transform('spherical')
class Spherical(BaseTransform):
    r"""Saves the spherical coordinates of linked nodes in its edge attributes
    (functional name: :obj:`spherical`).

    Args:
        norm (bool, optional): If set to :obj:`False`, the output will not be
            normalized to the interval :math:`{[0, 1]}^3`.
            (default: :obj:`True`)
        max_value (float, optional): If set and :obj:`norm=True`, normalization
            will be performed based on this value instead of the maximum value
            found in the data. (default: :obj:`None`)
        cat (bool, optional): If set to :obj:`False`, all existing edge
            attributes will be replaced. (default: :obj:`True`)
    """
    def __init__(
        self,
        norm: bool = True,
        max_value: Optional[float] = None,
        cat: bool = True,
    ):
        self.norm = norm
        self.max = max_value
        self.cat = cat

    def forward(self, data: Data) -> Data:
        (row, col), pos, pseudo = data.edge_index, data.pos, data.edge_attr
        assert pos.dim() == 2 and pos.size(1) == 3

        cart = pos[col] - pos[row]

        rho = torch.norm(cart, p=2, dim=-1).view(-1, 1)

        theta = torch.atan2(cart[..., 1], cart[..., 0]).view(-1, 1)
        theta = theta + (theta < 0).type_as(theta) * (2 * PI)

        phi = torch.acos(cart[..., 2] / rho.view(-1)).view(-1, 1)

        if self.norm:
            rho = rho / (rho.max() if self.max is None else self.max)
            theta = theta / (2 * PI)
            phi = phi / PI

        spher = torch.cat([rho, theta, phi], dim=-1)

        if pseudo is not None and self.cat:
            pseudo = pseudo.view(-1, 1) if pseudo.dim() == 1 else pseudo
            data.edge_attr = torch.cat([pseudo, spher.type_as(pos)], dim=-1)
        else:
            data.edge_attr = spher

        return data

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(norm={self.norm}, '
                f'max_value={self.max})')
