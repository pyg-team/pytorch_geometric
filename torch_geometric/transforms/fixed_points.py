from __future__ import division

import re
import math

import torch
import numpy as np


class FixedPoints(object):
    r"""Samples a fixed number of :obj:`num` points and features from a point
    cloud.

    Args:
        num (int): The number of points to sample.
        replace (bool, optional): If set to :obj:`False`, samples fixed
            points without replacement. In case :obj:`num` is greater than
            the number of points, duplicated points are kept to a
            minimum. (default: :obj:`True`)
    """

    def __init__(self, num, replace=True):
        self.num = num
        self.replace = replace

    def __call__(self, data):
        num_nodes = data.num_nodes

        if self.replace:
            choice = np.random.choice(num_nodes, self.num, replace=True)
        else:
            choice = torch.cat([
                torch.randperm(num_nodes)
                for _ in range(math.ceil(self.num / num_nodes))
            ], dim=0)[:self.num]

        for key, item in data:
            if bool(re.search('edge', key)):
                continue
            if torch.is_tensor(item) and item.size(0) == num_nodes:
                data[key] = item[choice]

        return data

    def __repr__(self):
        return '{}({}, replace={})'.format(self.__class__.__name__, self.num,
                                           self.replace)
