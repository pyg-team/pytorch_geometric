import torch
from torch_unique import unique


def _get_dtype(max_value):
    if max_value <= 255:
        return torch.uint8
    elif max_value <= 32767:  # pragma: no cover
        return torch.short
    elif max_value <= 2147483647:  # pragma: no cover
        return torch.int
    else:  # pragma: no cover
        return torch.long


def consecutive_cluster(src):
    size = src.size(0)
    key, perm = unique(src)
    max_value = key.size(0)
    dtype = _get_dtype(max_value)
    arg = torch.empty((key[-1] + 1, ), dtype=dtype, device=src.device)
    arg[key] = torch.arange(0, max_value, dtype=dtype, device=src.device)
    out = arg[src.view(-1)]
    out = out.view(size).long()
    return out, perm
