import torch


def identity(size):
    if len(size) != 2:
        raise ValueError('Invalid sparse dimensionality')

    y, x = size

    values = torch.ones(min(y, x))
    indices = torch.arange(0, min(y, x)).long()
    indices = indices.repeat(2, 1)

    return torch.sparse.FloatTensor(indices, values, size)
