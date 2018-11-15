import torch
from torch_geometric.utils import degree, one_hot


class OneHotDegree(object):
    r"""Adds the node degree as one hot encodings to the node features.

    Args:
        max_degree (int): Maximum degree.
        cat (bool, optional): Concat node degrees to node features instead
            of replacing them. (default: :obj:`True`)

    .. testsetup::

        import torch
        from torch_geometric.data import Data

    .. testcode::

        from torch_geometric.transforms import OneHotDegree

        edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
        data = Data(edge_index=edge_index)

        data = OneHotDegree(max_degree=2)(data)

        print(data.x)

    .. testoutput::

        tensor([[0., 1., 0.],
                [0., 0., 1.],
                [0., 1., 0.]])
    """

    def __init__(self, max_degree, cat=True):
        self.max_degree = max_degree
        self.cat = cat

    def __call__(self, data):
        row, x = data.edge_index[0], data.x
        deg = degree(row, data.num_nodes)
        deg = one_hot(deg, num_classes=self.max_degree + 1)

        if x is not None and self.cat:
            x = x.view(-1, 1) if x.dim() == 1 else x
            data.x = torch.cat([x, deg.to(x.dtype)], dim=-1)
        else:
            data.x = deg

        return data

    def __repr__(self):
        return '{}({}, cat={})'.format(self.__class__.__name__,
                                       self.max_degree, self.cat)
