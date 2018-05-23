import random


class RandomFlip(object):
    """Flips node positions along a given axis randomly with a given
    probability.

    Args:
        axis (int): The axis along the position of nodes being flipped.
        p (float, optional): Probability of the position of nodes being
            flipped. (default: :obj:`0.5`)

    .. testsetup::

        import torch
        from torch_geometric.data import Data

    .. testcode::

        from torch_geometric.transforms import RandomFlip

        pos = torch.tensor([[-1, 1], [-3, 0], [2, -1]], dtype=torch.float)
        data = Data(pos=pos)

        data = RandomFlip(axis=0, p=1)(data)

        print(data.pos)

    .. testoutput::

        tensor([[ 1.,  1.],
                [ 3.,  0.],
                [-2., -1.]])
    """

    def __init__(self, axis, p=0.5):
        self.axis = axis
        self.p = p

    def __call__(self, data):
        if random.random() < self.p:
            data.pos[:, self.axis] = -data.pos[:, self.axis]
        return data

    def __repr__(self):
        return '{}(axis={}, p={})'.format(self.__class__.__name__, self.axis,
                                          self.p)
