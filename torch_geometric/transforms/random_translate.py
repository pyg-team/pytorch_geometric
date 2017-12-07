import torch


class RandomTranslate(object):
    def __init__(self, max):
        self.max = abs(max)

    def __call__(self, data):
        position = data.position
        translate = torch.rand(position.size()) * 2 * self.max - self.max
        position += translate.type_as(position)
        return data
