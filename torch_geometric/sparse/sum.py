import torch


def sum(a, dim=None):
    if dim is None:
        return torch.sum(a._values())

    value = a._values()
    index = a._indices()[1 - dim]
    n = a.size(1 - dim)

    return value.new(n).fill_(0).scatter_add_(0, index, value)
