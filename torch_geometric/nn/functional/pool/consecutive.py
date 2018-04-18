import torch
from torch_unique import unique_by_key


def _get_type(max_value, cuda):
    t = torch.cuda if cuda else torch

    if max_value <= 255:
        return t.ByteTensor
    elif max_value <= 32767:  # pragma: no cover
        return t.ShortTensor
    elif max_value <= 2147483647:  # pragma: no cover
        return t.IntTensor
    else:  # pragma: no cover
        return t.LongTensor


def consecutive_cluster(src, batch=None):
    size = src.size(0)
    key, value = unique_by_key(src, torch.arange(0, size, out=src.new()))
    len = key[-1] + 1
    max_value = key.size(0)
    type = _get_type(max_value, src.is_cuda)
    arg = type(len)
    arg[key] = torch.arange(0, max_value, out=type(max_value))
    output = arg[src.view(-1)]
    output = output.view(size).long()

    return (output, None) if batch is None else (output, batch[value])
