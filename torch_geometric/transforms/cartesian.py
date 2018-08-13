import torch


class Cartesian(object):
    r"""Saves the globally normalized spatial relation of linked nodes as
    Cartesian coordinates (mapped to the fixed interval :math:`[0, 1]`)

    .. math::
        \mathbf{u}(i,j) = 0.5 + \frac{\mathbf{pos}_j - \mathbf{pos}_i}{2 \cdot
        \max_{(v, w) \in \mathcal{E}} | \mathbf{pos}_w - \mathbf{pos}_v|}

    in its edge attributes.

    Args:
        cat (bool, optional): Concat pseudo-coordinates to edge attributes
            instead of replacing them. (default: :obj:`True`)

    .. testsetup::

        import torch
        from torch_geometric.data import Data

    .. testcode::

        from torch_geometric.transforms import Cartesian

        pos = torch.tensor([[-1, 0], [0, 0], [2, 0]], dtype=torch.float)
        edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
        data = Data(edge_index=edge_index, pos=pos)

        data = Cartesian()(data)

        print(data.edge_attr)

    .. testoutput::

        tensor([[0.7500, 0.5000],
                [0.2500, 0.5000],
                [1.0000, 0.5000],
                [0.0000, 0.5000]])
    """

    def __init__(self, cat=True):
        self.cat = cat

    def __call__(self, data):
        (row, col), pos, pseudo = data.edge_index, data.pos, data.edge_attr

        cart = pos[col] - pos[row]
        cart = cart / (2 * cart.abs().max())
        cart += 0.5
        cart = cart.view(-1, 1) if cart.dim() == 1 else cart

        if pseudo is not None and self.cat:
            pseudo = pseudo.view(-1, 1) if pseudo.dim() == 1 else pseudo
            data.edge_attr = torch.cat([pseudo, cart.type_as(pseudo)], dim=-1)
        else:
            data.edge_attr = cart

        return data

    def __repr__(self):
        return '{}(cat={})'.format(self.__class__.__name__, self.cat)
