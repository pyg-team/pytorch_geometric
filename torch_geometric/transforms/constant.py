import torch


class Constant(object):
    r"""Adds a constant value to each node feature.

    Args:
        value (int, optional): The value to add. (default: :obj:`1`)
        cat (bool, optional): If set to :obj:`False`, all existing node
            features will be replaced. (default: :obj:`True`)
    """
    def __init__(self, value=1, cat=True):
        self.value = value
        self.cat = cat

    def __call__(self, data):
        x = data.x

        c = torch.full((data.num_nodes, 1), self.value, dtype=torch.float)

        if x is not None and self.cat:
            x = x.view(-1, 1) if x.dim() == 1 else x
            data.x = torch.cat([x, c.to(x.dtype).to(x.device)], dim=-1)
        else:
            data.x = c

        return data

    def __repr__(self):
        return '{}(value={})'.format(self.__class__.__name__, self.value)
