import torch


class RandomScale(object):
    def __init__(self, scale):
        self.scale = abs(scale)

    def __call__(self, data):
        position = data.position
        mean = position.mean(dim=0)
        position -= mean
        print(position)

        scale = position.new(position.size(1)).uniform_(
            -self.scale, self.scale)
        # scale = (scale + 1).diag()
        print(scale)

        position *= scale
        print(position)
        position += mean
        print(position)

        # translate = position.new(position.size()).uniform_(-self.std, self.std)
        # data.position = position + translate
        return data
