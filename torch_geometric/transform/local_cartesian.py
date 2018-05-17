import torch
from torch_scatter import scatter_max


class LocalCartesian(object):
    r"""Saves the locally normalized spatial relation of linked nodes as
    Cartesian coordinates (mapped to the fixed interval :math:`[0, 1]`)

    .. math::
        \mathbf{u}(i,j) = 0.5 + \frac{\mathbf{pos}_j - \mathbf{pos}_i}{2 \cdot
        \max_{v \in \mathcal{N}(i)} | \mathbf{pos}_v - \mathbf{pos}_i|}

    in its edge attributes.

    Args:
        cat (bool, optional): Concat pseudo-coordinates to edge attributes
            instead of replacing them. (default: :obj:`True`)

    .. testsetup::

        import torch
        from torch_geometric.datasets.dataset import Data

    .. testcode::

        from torch_geometric.transform import LocalCartesian

        pos = torch.tensor([[-1, 0], [0, 0], [2, 0]], dtype=torch.float)
        edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
        data = Data(None, pos, edge_index, None, None)

        data = LocalCartesian()(data)

        print(data.weight)

    .. testoutput::

        tensor([[ 1.0000,  0.5000],
                [ 0.2500,  0.5000],
                [ 1.0000,  0.5000],
                [ 0.0000,  0.5000]])
    """

    def __init__(self, cat=True):
        self.cat = cat

    def __call__(self, data):
        (row, col), pos, pseudo = data.index, data.pos, data.weight

        cart = pos[col] - pos[row]
        max_cart, _ = scatter_max(cart.abs(), row, 0, dim_size=pos.size(0))
        cart = cart / (2 * max_cart.max(dim=1, keepdim=True)[0][row])
        cart += 0.5
        cart = cart.view(-1, 1) if cart.dim() == 1 else cart

        if pseudo is not None and self.cat:
            pseudo = pseudo.view(-1, 1) if pseudo.dim() == 1 else pseudo
            data.weight = torch.cat([pseudo, cart.type_as(pseudo)], dim=-1)
        else:
            data.weight = cart

        return data

    def __repr__(self):
        return '{}(cat={})'.format(self.__class__.__name__, self.cat)
