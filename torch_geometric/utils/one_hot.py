import torch

from .repeat import repeat


def one_hot(src, num_classes=None, dtype=None):
    r"""Converts labels into a one-hot format.

    Args:
        src (Tensor): The labels.
        num_classes (int or list, optional): The number of classes.
            (default: :obj:`None`)
        dtype (:obj:`torch.dtype`, optional): The desired data type of the
            returned tensor.

    :rtype: :class:`Tensor`
    """

    src = src.to(torch.long)
    src = src.unsqueeze(-1) if src.dim() == 1 else src
    assert src.dim() == 2

    if num_classes is None:
        num_classes = src.max(dim=0)[0] + 1
    else:
        if torch.is_tensor(num_classes):
            num_classes = num_classes.tolist()

        num_classes = torch.tensor(
            repeat(num_classes, length=src.size(1)),
            dtype=torch.long,
            device=src.device)

    if src.size(1) > 1:
        zero = torch.tensor([0], device=src.device)
        src = src + torch.cat([zero, torch.cumsum(num_classes, 0)[:-1]])

    size = src.size(0), num_classes.sum()
    out = torch.zeros(size, dtype=dtype, device=src.device)
    out.scatter_(1, src, 1)
    return out
