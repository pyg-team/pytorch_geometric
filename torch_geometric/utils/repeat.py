import itertools
import numbers

import torch


def repeat(src, length):
    if src is None:
        return None

    if torch.is_tensor(src):
        if src.numel() == 1:
            return src.repeat(length)

        current_length = src.size(0)

        if current_length > length:
            return src[:length]
        if current_length < length:
            last_element = src[-1].unsqueeze(0)
            padding_length = length - current_length
            padding = last_element.repeat(padding_length)
            src = torch.cat((src, padding))
    elif isinstance(src, numbers.Number):
        return list(itertools.repeat(src, length))
    else:
        current_length = len(src)
        if (current_length > length):
            return src[:length]
        if (current_length < length):
            return src + list(
                itertools.repeat(src[-1], length - current_length))

    return src
