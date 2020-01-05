import random


class RandomFlip(object):
    """Flips node positions along a given axis randomly with a given
    probability.

    Args:
        axis (int): The axis along the position of nodes being flipped.
        p (float, optional): Probability that node positions will be flipped.
            (default: :obj:`0.5`)
    """

    def __init__(self, axis, p=0.5):
        self.axis = axis
        self.p = p

    def __call__(self, data):
        if random.random() < self.p:
            pos = data.pos.clone()
            pos[..., self.axis] = -pos[..., self.axis]
            data.pos = pos
        return data

    def __repr__(self):
        return '{}(axis={}, p={})'.format(self.__class__.__name__, self.axis,
                                          self.p)
