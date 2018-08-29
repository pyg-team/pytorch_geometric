import torch


class Distance(object):
    r"""Saves the Euclidean distance of linked nodes in its edge attributes.

    Args:
        cat (bool, optional): Concat pseudo-coordinates to edge attributes
            instead of replacing them. (default: :obj:`True`)

    .. testsetup::

        import torch
        from torch_geometric.data import Data

    .. testcode::

        from torch_geometric.transforms import Distance

        pos = torch.tensor([[-1, 0], [0, 0], [2, 0]], dtype=torch.float)
        edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
        data = Data(edge_index=edge_index, pos=pos)

        data = Distance()(data)

        print(data.edge_attr)

    .. testoutput::

        tensor([[1.],
                [1.],
                [2.],
                [2.]])
    """

    def __init__(self, cat=True):
        self.cat = cat

    def __call__(self, data):
        (row, col), pos, pseudo = data.edge_index, data.pos, data.edge_attr

        dist = torch.norm(pos[col] - pos[row], p=2, dim=-1).view(-1, 1)

        if pseudo is not None and self.cat:
            pseudo = pseudo.view(-1, 1) if pseudo.dim() == 1 else pseudo
            data.edge_attr = torch.cat([pseudo, dist.type_as(pseudo)], dim=-1)
        else:
            data.edge_attr = dist

        return data

    def __repr__(self):
        return '{}(cat={})'.format(self.__class__.__name__, self.cat)
