import torch


def mm_diagonal(a, b, transpose=False):
    index = b._indices()
    target = index[1] if transpose else index[0]
    value = b._values()

    value = a[target] * value

    return torch.sparse.FloatTensor(index, value, b.size())
