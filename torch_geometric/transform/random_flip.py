import random


def flip(data, axis):
    pos = data.pos

    min, max = pos.min(0)[0], pos.max(0)[0]
    offset = min + 0.5 * (max - min)
    scale = pos.new(pos.size(1)).fill_(1)
    scale[axis] = -1

    pos = pos - offset  # Translate to origin.
    pos *= scale  # Flip.
    pos += offset  # Translate back.

    data.pos = pos
    return data


class RandomFlip(object):
    def __init__(self, axis, p=0.5):
        self.axis = axis
        self.p = p

    def __call__(self, data):
        if random.random() < self.p:
            return flip(data, self.axis)
        return data
