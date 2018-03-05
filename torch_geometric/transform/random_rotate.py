from math import sin, cos

import torch


class RandomRotate(object):
    def __init__(self, max):
        self.max = abs(max)

    def __call__(self, data):
        mean = data.pos.mean(dim=0)
        pos = data.pos - mean

        a = 2 * self.max * torch.rand(1)[0] - self.max
        s, c = sin(a), cos(a)
        dim = pos.size(1)
        if dim == 2:
            rotate = torch.FloatTensor([[c, -s], [s, c]])
        elif dim == 3:
            rotate = torch.FloatTensor([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        else:
            raise ValueError

        rotate = rotate.type_as(pos)
        pos = torch.matmul(rotate, pos.view(-1, dim, 1))
        pos = pos.view(-1, dim)

        pos += mean
        data.pos = pos
        return data
