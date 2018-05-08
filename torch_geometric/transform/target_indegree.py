import torch
from torch_geometric.utils import degree


class TargetIndegree(object):
    r"""Saves the globally normalized degree of target nodes

    .. math::

        \mathbf{u}(i,j) = \frac{\deg(j)}{\max_{v \in \mathcal{V}} \deg(v)}

    as edge attributes.

    Args:
        cat (bool, optional): Concat pseudo-coordinates to edge attributes
            instead of replacing them. (default: :obj:`True`)

    .. testsetup::

        import torch
        from torch_geometric.datasets.dataset import Data

    .. testcode::

        from torch_geometric.transform import TargetIndegree

        edge_index = torch.LongTensor([[0, 1, 1, 2], [1, 0, 2, 1]])
        data = Data(None, None, edge_index, None, None)

        data = TargetIndegree()(data)

        print(data.weight)

    .. testoutput::

        tensor([ 1.0000,  0.5000,  0.5000,  1.0000])
    """

    def __init__(self, cat=True):
        self.cat = cat

    def __call__(self, data):
        col, pseudo = data.index[1], data.weight

        deg = degree(col, data.num_nodes)
        deg /= deg.max()
        deg = deg[col]

        if pseudo is not None and self.cat:
            pseudo = pseudo.view(-1, 1) if pseudo.dim() == 1 else pseudo
            deg = deg.type_as(pseudo)
            data.weight = torch.cat([pseudo, deg], dim=-1)
        else:
            data.weight = deg

        return data

    def __repr__(self):
        return '{}(cat={})'.format(self.__class__.__name__, self.cat)
