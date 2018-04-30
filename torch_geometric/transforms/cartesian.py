import torch


class Cartesian(object):
    r"""Saves the globally normalized spatial relation of linked nodes in
    Cartesian coordinates

    .. math::
        \mathbf{u}(i,j) = 0.5 + \frac{\mathbf{pos}_j - \mathbf{pos}_i}{2 \cdot
        \max_{(v, w) \in \mathcal{E}} | \mathbf{pos}_w - \mathbf{pos}_v|}

    as edge attributes.

    Args:
        cat (bool, optional): Concat pseudo-coordinates to edge attributes
            instead of replacing them. (default: :obj:`True`)

    .. testsetup::

        import torch
        from torch_geometric.data import Data

    .. testcode::

        from torch_geometric.transforms import Cartesian

        pos = torch.Tensor([[-1, 0], [0, 0], [2, 0]])
        edge_index = torch.LongTensor([[0, 1, 1, 2], [1, 0, 2, 1]])
        data = Data(edge_index=edge_index, pos=pos)

        data = Cartesian()(data)

        print(data.edge_attr)

    .. testoutput::

        tensor([[ 0.7500,  0.5000],
                [ 0.2500,  0.5000],
                [ 1.0000,  0.5000],
                [ 0.0000,  0.5000]])
    """

    def __init__(self, cat=True):
        self.cat = cat

    def __call__(self, data):
        (row, col), pos, pseudo = data.edge_index, data.pos, data.edge_attr

        cartesian = pos[col] - pos[row]
        cartesian /= 2 * cartesian.abs().max()
        cartesian += 0.5

        if pseudo is not None and self.cat:
            pseudo = pseudo.unsqueeze(-1) if pseudo.dim() == 1 else pseudo
            cartesian = cartesian.type_as(pseudo)
            data.edge_attr = torch.cat([pseudo, cartesian], dim=-1)
        else:
            data.edge_attr = cartesian

        return data

    def __repr__(self):
        return '{}(cat={})'.format(self.__class__.__name__, self.cat)
