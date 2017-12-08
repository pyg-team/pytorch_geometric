from __future__ import division

import torch


class RandomScale(object):
    def __init__(self, scale):
        self.max = max(abs(scale), 1)

    def __call__(self, data):
        position = data.position
        mean = position.mean(dim=0)
        position -= mean

        scale = 2 * torch.rand(position.size(1))
        upscale_mask = scale > 1
        downscale_mask = scale < 1

        scale[upscale_mask] -= 1
        scale[upscale_mask] *= (self.max - 1)
        scale[upscale_mask] += 1

        inv = 1 / self.max
        scale[downscale_mask] *= (1 - inv)
        scale[downscale_mask] += inv

        position *= scale.type_as(position)

        position += mean
        return data
