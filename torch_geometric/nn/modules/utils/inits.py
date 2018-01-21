from __future__ import division

import math


def uniform(size, tensor):
    stdv = 1 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-stdv, stdv)
