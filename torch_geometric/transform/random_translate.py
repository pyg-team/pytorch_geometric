import torch


class RandomTranslate(object):
    def __init__(self, max):
        self.max = abs(max)

    def __call__(self, data):
        translate = torch.rand(data.pos.size()) * 2 * self.max - self.max
        data.pos = data.pos + translate.type_as(data.pos)
        return data
