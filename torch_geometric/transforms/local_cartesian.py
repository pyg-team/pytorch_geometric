import torch
from torch_scatter import scatter_max


class LocalCartesian(object):
    r"""Saves the relative Cartesian coordinates of linked nodes in its edge
    attributes. Each coordinate gets *neighborhood-normalized* to the
    interval :math:`{[0, 1]}^D`.

    Args:
        cat (bool, optional): If set to :obj:`False`, all existing edge
            attributes will be replaced. (default: :obj:`True`)
    """

    def __init__(self, cat=True):
        self.cat = cat

    def __call__(self, data):
        (row, col), pos, pseudo = data.edge_index, data.pos, data.edge_attr

        cart = pos[col] - pos[row]
        cart = cart.view(-1, 1) if cart.dim() == 1 else cart

        max_value, _ = scatter_max(cart.abs(), row, 0, dim_size=pos.size(0))
        max_value = max_value.max(dim=-1, keepdim=True)[0]
        cart = cart / (2 * max_value[row]) + 0.5

        if pseudo is not None and self.cat:
            pseudo = pseudo.view(-1, 1) if pseudo.dim() == 1 else pseudo
            data.edge_attr = torch.cat([pseudo, cart.type_as(pseudo)], dim=-1)
        else:
            data.edge_attr = cart

        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)
