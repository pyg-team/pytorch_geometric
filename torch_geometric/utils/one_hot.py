import torch


def one_hot(src, num_classes=None, dtype=None, device=None):
    num_classes = src.max().item() + 1 if num_classes is None else num_classes
    out = torch.zeros(src.size(0), num_classes, dtype=dtype, device=device)
    out.scatter_(1, src.unsqueeze(-1).expand_as(out), 1)
    return out
