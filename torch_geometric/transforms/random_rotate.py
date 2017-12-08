from __future__ import division

from math import sin, cos

import torch


class RandomRotate(object):
    def __init__(self, max):
        self.max = abs(max)

    def __call__(self, data):
        position = data.position
        mean = position.mean(dim=0)
        position -= mean

        a = 2 * self.max * torch.rand(1)[0] - self.max
        s, c = sin(a), cos(a)
        dim = position.size(1)
        if dim == 2:
            rotate = torch.FloatTensor([[c, -s], [s, c]])
        elif dim == 3:
            rotate = torch.FloatTensor([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        else:
            raise ValueError

        rotate = rotate.type_as(position)
        position = torch.matmul(rotate, position.view(-1, dim, 1))
        position = position.view(-1, dim)

        position += mean
        data.position = position
        return data
