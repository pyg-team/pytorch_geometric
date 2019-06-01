import torch
import numpy as np


class FixedPoints(object):
    r"""Samples a fixed number of :obj:`num` points and features from a point
    cloud.

    Args:
        num (int): The number of points to sample.
    """

    def __init__(self, num):
        self.num = num

    def __call__(self, data):
        num_nodes = data.num_nodes
        choice = np.random.choice(data.num_nodes, self.num, replace=True)

        for key, item in data:
            if torch.is_tensor(item) and item.size(0) == num_nodes:
                data[key] = item[choice]

        return data

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.num)
