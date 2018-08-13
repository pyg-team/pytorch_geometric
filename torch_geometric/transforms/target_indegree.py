import torch
from torch_geometric.utils import degree


class TargetIndegree(object):
    r"""Saves the globally normalized degree of target nodes (mapped to the
    fixed interval :math:`[0, 1]`)

    .. math::

        \mathbf{u}(i,j) = \frac{\deg(j)}{\max_{v \in \mathcal{V}} \deg(v)}

    in its edge attributes.

    Args:
        cat (bool, optional): Concat pseudo-coordinates to edge attributes
            instead of replacing them. (default: :obj:`True`)

    .. testsetup::

        import torch
        from torch_geometric.data import Data

    .. testcode::

        from torch_geometric.transforms import TargetIndegree

        edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
        data = Data(edge_index=edge_index)

        data = TargetIndegree()(data)

        print(data.edge_attr)

    .. testoutput::

        tensor([[1.0000],
                [0.5000],
                [0.5000],
                [1.0000]])
    """

    def __init__(self, cat=True):
        self.cat = cat

    def __call__(self, data):
        col, pseudo = data.edge_index[1], data.edge_attr

        deg = degree(col, data.num_nodes)
        deg = deg / deg.max()
        deg = deg[col]
        deg = deg.view(-1, 1)

        if pseudo is not None and self.cat:
            pseudo = pseudo.view(-1, 1) if pseudo.dim() == 1 else pseudo
            data.edge_attr = torch.cat([pseudo, deg.type_as(pseudo)], dim=-1)
        else:
            data.edge_attr = deg

        return data

    def __repr__(self):
        return '{}(cat={})'.format(self.__class__.__name__, self.cat)
