from typing import Optional

import torch
from torch import Tensor


def one_hot(
    index: Tensor,
    num_classes: Optional[int] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    r"""Taskes a one-dimensional :obj:`index` tensor and returns a one-hot
    encoded representation of it with shape :obj:`[*, num_classes]` that has
    zeros everywhere except where the index of last dimension matches the
    corresponding value of the input tensor, in which case it will be :obj:`1`.

    .. note::
        This is a more memory-efficient version of
        :meth:`torch.nn.functional.one_hot` as you can customize the output
        :obj:`dtype`.

    Args:
        index (torch.Tensor): The one-dimensional input tensor.
        num_classes (int, optional): The total number of classes. If set to
            :obj:`None`, the number of classes will be inferred as one greater
            than the largest class value in the input tensor.
            (default: :obj:`None`)
        dtype (torch.dtype, optional): The :obj:`dtype` of the output tensor.
    """
    if index.dim() != 1:
        raise ValueError("'index' tensor needs to be one-dimensional")

    if num_classes is None:
        num_classes = int(index.max()) + 1

    out = torch.zeros((index.size(0), num_classes), dtype=dtype,
                      device=index.device)
    return out.scatter_(1, index.unsqueeze(1), 1)
