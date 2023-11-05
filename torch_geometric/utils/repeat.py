import itertools
import numbers
from typing import Any

import torch
from torch import Tensor


def repeat(src: Any, length: int) -> Any:
    if src is None:
        return None

    if isinstance(src, Tensor):
        if src.numel() == 1:
            return src.repeat(length)

        if src.numel() > length:
            return src[:length]

        if src.numel() < length:
            last_elem = src[-1].unsqueeze(0)
            padding = last_elem.repeat(length - src.numel())
            return torch.cat([src, padding])

        return src

    if isinstance(src, numbers.Number):
        return list(itertools.repeat(src, length))

    if (len(src) > length):
        return src[:length]

    if (len(src) < length):
        return src + list(itertools.repeat(src[-1], length - len(src)))

    return src
