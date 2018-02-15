import torch


class RandomShear(object):
    def __init__(self, max):
        self.max = abs(max)

    def __call__(self, data):
        mean = data.pos.mean(dim=0)
        pos = data.pos - mean

        dim = pos.size(1)
        shear = torch.rand(dim, dim) * 2 * self.max - self.max
        range = torch.arange(0, dim, out=torch.LongTensor())
        shear[range, range] = 1

        shear = shear.type_as(pos)
        pos = torch.matmul(shear, pos.view(-1, dim, 1))
        pos = pos.view(-1, dim)

        pos += mean
        data.pos = pos
        return data
