from functools import reduce
from itertools import product

import torch


def _spline_cpu(input, kernel_size, is_open_spline, degree):
    """
    Args:
        input (Tensor): 1d or 2d tensor.
        kernel_size (list)
        is_open_spline (list)
        spline_degree (int, optional): B-Spline degree. (default: 1)
    """

    if degree != 1:
        raise NotImplementedError()

    input = input.unsqueeze(1) if len(input.size()) < 2 else input
    input = input * (kernel_size - is_open_spline).type_as(input)

    amount = input.frac()
    amount = torch.stack([amount, 1 - amount], dim=len(input.size()))

    bot = input.floor().long()
    top = (bot + 1) % kernel_size
    bot %= kernel_size
    index = torch.stack([top, bot], dim=len(input.size()))

    return amount, index


def _create_mask(dim, m, type=torch.LongTensor):
    mask = list(product(*[range(m) for _ in range(dim)]))
    mask = torch.LongTensor(mask).type(type)
    mask += torch.arange(0, dim * m, m).type_as(mask)
    return mask


def spline_cpu(input, kernel_size, is_open_spline, degree):
    amount, index = _spline_cpu(input, kernel_size, is_open_spline, degree)

    dim = amount.size(1)
    m = amount.size(2)

    mask = _create_mask(dim, m, index.type())

    amount = amount.view(-1, m * dim)
    amount = amount[:, mask.view(-1)]
    amount = amount.view(-1, m**dim, dim)
    amount = amount.prod(2)

    off = [reduce(lambda x, y: x * y, kernel_size[i:]) for i in range(1, dim)]
    off.append(1)
    off = torch.LongTensor([off]).type_as(index).t()
    index = off * index

    index = index.view(-1, m * dim)
    index = index[:, mask.view(-1)]
    index = index.view(-1, m**dim, dim)
    index = index.sum(2)

    return amount, index
