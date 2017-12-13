import torch


class RandomShear(object):
    def __init__(self, max):
        self.max = abs(max)

    def __call__(self, data):
        position = data.position
        mean = position.mean(dim=0)
        position -= mean

        dim = position.size(1)
        shear = torch.rand(dim, dim) * 2 * self.max - self.max
        range = torch.arange(0, dim, out=torch.LongTensor())
        shear[range, range] = 1

        shear = shear.type_as(position)
        position = torch.matmul(shear, position.view(-1, dim, 1))
        position = position.view(-1, dim)

        position += mean
        data.position = position
        return data
