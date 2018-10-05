import torch


def one_hot(src, num_classes=None, dtype=None):
    src = src.to(torch.long)
    src = src.unsqueeze(-1) if src.dim() == 1 else src
    assert src.dim() == 2

    if num_classes is None:
        num_classes = src.max(dim=0)[0] + 1
    elif isinstance(num_classes, int):
        num_classes = torch.tensor(num_classes)

    if src.size(1) > 1:
        zero = torch.tensor([0], device=src.device)
        src = src + torch.cat([zero, torch.cumsum(num_classes, 0)[:-1]])

    size = src.size(0), num_classes.sum()
    out = torch.zeros(size, dtype=dtype, device=src.device)
    out.scatter_(1, src, 1)
    return out
