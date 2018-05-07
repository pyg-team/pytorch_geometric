import torch


def one_hot(src, size=None, dtype=None, device=None):
    size = src.max().item() + 1 if size is None else size

    out = torch.zeros(src.size(0), size, dtype=dtype, device=device)
    out.scatter_(1, src.unsqueeze(-1).expand_as(out), 1)
    return out
