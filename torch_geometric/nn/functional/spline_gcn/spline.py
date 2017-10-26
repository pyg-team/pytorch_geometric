from functools import reduce
from itertools import product

import torch


def create_mask(dim, m, type=torch.LongTensor):
    mask = list(product(*[range(m) for _ in range(dim)]))
    mask = torch.LongTensor(mask).type(type)
    mask += torch.arange(0, dim * m, m).type_as(mask)
    return mask


def spline(values, kernel_size, is_open_spline, degree):
    """
    Args:
        values (Tensor): 1d or 2d tensor.
        kernel_size (list)
        is_open_spline (list)
        spline_degree (int, optional): B-Spline degree. (default: 1)
    """

    if degree != 1:
        raise NotImplementedError()

    kernel_size = kernel_size - is_open_spline

    values = values.unsqueeze(1) if len(values.size()) < 2 else values
    values = values * kernel_size.type_as(values)

    amount = values.frac()
    amount = torch.stack([amount, 1 - amount], dim=len(values.size()))

    index_g = values.floor().long()

    # Open splines adjustments.
    kernel_size = kernel_size.type_as(index_g)
    is_open_spline = is_open_spline.type_as(index_g)
    index_g -= is_open_spline * (index_g >= kernel_size).type_as(index_g)
    index_f = index_g + is_open_spline

    # Closed splines adjustments.
    is_closed_spline = 1 - is_open_spline
    index_g -= is_closed_spline
    index_g += is_closed_spline * (index_g < 0).type_as(index_g) * kernel_size
    index_f -= is_closed_spline * (
        index_f >= kernel_size).type_as(index_f) * kernel_size

    index = torch.stack([index_f, index_g], dim=len(values.size()))

    return amount, index


def spline_weights(values, kernel_size, is_open_spline, degree):
    amount, index = spline(values, kernel_size, is_open_spline, degree)

    dim = amount.size(1)
    m = amount.size(2)

    mask = create_mask(dim, m, index.type())

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
