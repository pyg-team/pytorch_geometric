import torch


class Constant(object):
    r"""Adds a constant value to each node feature.

    Args:
        value (int): The value to add.
        cat (bool, optional): Concat value to node features instead
            of replacing them. (default: :obj:`True`)

    .. testsetup::

        import torch
        from torch_geometric.data import Data

    .. testcode::

        from torch_geometric.transforms import Constant

        edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
        data = Data(edge_index=edge_index)

        data = Constant(value=1)(data)

        print(data.x)

    .. testoutput::

        tensor([[1.],
                [1.],
                [1.]])
    """

    def __init__(self, value, cat=True):
        self.value = value
        self.cat = cat

    def __call__(self, data):
        x = data.x

        c = torch.full((data.num_nodes, 1), self.value)

        if x is not None and self.cat:
            x = x.view(-1, 1) if x.dim() == 1 else x
            data.x = torch.cat([x, c.to(x.dtype).to(x.device)], dim=-1)
        else:
            data.x = c

        return data

    def __repr__(self):
        return '{}({}, cat={})'.format(self.__class__.__name__, self.value,
                                       self.cat)
