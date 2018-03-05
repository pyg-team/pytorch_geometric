from __future__ import division

import torch


class RandomScale(object):
    def __init__(self, scale):
        self.max = max(abs(scale), 1)

    def __call__(self, data):
        mean = data.pos.mean(dim=0)
        pos = data.pos - mean

        scale = 2 * torch.rand(pos.size(1))
        upscale_mask = scale > 1
        downscale_mask = scale < 1

        scale[upscale_mask] -= 1
        scale[upscale_mask] *= (self.max - 1)
        scale[upscale_mask] += 1

        inv = 1 / self.max
        scale[downscale_mask] *= (1 - inv)
        scale[downscale_mask] += inv

        pos *= scale.type_as(pos)

        pos += mean
        data.pos = pos
        return data
