import torch


class RandomScale(object):
    def __init__(self, max):
        self.max = abs(max)

    def __call__(self, data):
        position = data.position
        mean = position.mean(dim=0)
        position -= mean

        scale = torch.rand(position.size(1)) * 2 * self.max
        # scale = (scale + 1).diag()
        # print(scale)

        position *= scale
        # print(position)
        position += mean
        # print(position)

        # translate = position.new(position.size()).uniform_(-self.std, self.std)
        # data.position = position + translate
        return data
