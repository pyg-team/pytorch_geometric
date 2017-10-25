from functools import reduce
import itertools

import torch


def splines(values, kernel_size, is_open_spline, degree):
    """
    Args:
        values (Tensor): 1d or 2d tensor.
        kernel_size (list)
        is_open_spline (list)
        spline_degree (int, optional): B-Spline degree. (default: 1)
    """

    if degree != 1:
        raise NotImplementedError()

    values = values.unsqueeze(1) if len(values.size()) < 2 else values
    values *= (kernel_size - is_open_spline).type_as(values)

    amount = values.frac()
    amount = torch.stack([amount, 1 - amount], dim=len(values.size()))

    index = values.floor().long()
    index = index - is_open_spline * (index >= kernel_size).type_as(index)
    # index_1 = index_2 + 1

    index = torch.stack([index, index], dim=len(values.size()))

    return amount, index


def open_spline_amount(values, kernel_size, degree=1):
    # Passed `values` must be in the range [0, 1].
    kernel_size -= 1
    values = values * kernel_size
    amount_grow = values.frac()
    amount_fall = 1 - amount_grow
    return torch.stack([amount_grow, amount_fall], dim=len(values.size()))


def open_spline_index(values, kernel_size, degree=1):
    # Passed `values` must be in the range [0, 1].
    kernel_size -= 1
    values = values * kernel_size
    idx_fall = values.floor().long()
    idx_fall = idx_fall - (idx_fall >= kernel_size).long()
    idx_grow = 1 + idx_fall
    return torch.stack([idx_grow, idx_fall], dim=len(values.size()))


def closed_spline_amount(values, kernel_size, degree=1):
    # Passed `values` must be in the range [0, 1].
    values = values * kernel_size
    amount_grow = values.frac()
    amount_fall = 1 - amount_grow
    return torch.stack([amount_grow, amount_fall], dim=len(values.size()))


def closed_spline_index(values, kernel_size, degree=1):
    # Passed `values` must be in the range [0, 1].
    values = values * kernel_size
    idx_grow = values.floor().long()
    idx_fall = idx_grow - 1
    idx_grow = idx_grow - (idx_grow >= kernel_size).long() * kernel_size
    idx_fall = idx_fall + (idx_fall < 0).long() * kernel_size
    return torch.stack([idx_grow, idx_fall], dim=len(values.size()))


def _create_mask(dim, degree):
    m = degree + 1
    mask = list(itertools.product(*[range(m) for _ in range(dim)]))
    mask = torch.LongTensor(mask)
    mask += torch.arange(0, dim * m, m).long()
    mask = mask.cuda() if torch.cuda.is_available() else mask
    return mask


def weight_amount(values, kernel_size, is_open_spline, degree=1):

    return

    # torch.split(values, )

    dim = values.size(1)
    m = degree + 1

    # Collect all spline amounts for all dimensions with final shape
    # [|E| x dim x m]. Dimension 0 needs to be computed by using open splines,
    # whereas all other dimensions should be handled by closed splines.

    amount_0 = open_spline_amount(values[:, :1], degree)
    amount_1 = closed_spline_amount(values[:, 1:], degree)
    amount = torch.cat([amount_0, amount_1], dim=1)

    # Finally, we can compute a flattened out version of the tensor
    # [|E| x dim x m] with the shape [|E| x m^dim], which contains the amounts
    # of all dimensions for all possible combinations of m^dim. We do this by
    # defining a mask of all possible combinations with shape [m^dim x dim].
    # The product along the dimensions then describes the amounts influencing
    # the end vertex of each passed edge.

    # 1. Create mask.
    mask = create_mask(dim, degree)

    # 2. Apply mask.
    amount = amount.view(-1, m * dim)
    amount = amount[:, mask.view(-1)]
    amount = amount.view(-1, m**dim, dim)

    # 3. Multiply up dimensions.
    amount = amount.prod(2)

    return amount


def weight_index(values, kernel_size, degree=1):
    dim = len(kernel_size)
    m = degree + 1

    # Collect all spline indices for all dimensions with final shape
    # [|E| x dim x m]. Dimension 0 needs to be computed by using open splines,
    # whereas all other dimensions should be handled by closed splines.

    index_0 = open_spline_index(values[:, :1], kernel_size[:1], degree)
    index_1 = closed_spline_index(values[:, 1:], kernel_size[1:], degree)
    index = torch.cat([index_0, index_1], dim=1)

    # Element-wise broadcast-multiply an offset to indices in each dimension,
    # which allows flattening the indices in a later stage.
    # The offset is defined by the `kernel_size`, e.g. in the 3D case with
    # kernel [k_1, k_2, k_3]` the offset is given by [k_2 * k_3, k_3, 1].

    # 1. Calculate offset.
    off = [reduce(lambda x, y: x * y, kernel_size[i:]) for i in range(1, dim)]
    off.append(1)
    off = torch.LongTensor([off]).type_as(index).t()

    # 2. Apply offset.
    index = off * index

    # Finally, we can compute a flattened out version of the tensor
    # [|E| x dim x m] with the shape [|E| x m^dim], which contains the summed
    # up indices of all dimensions for all possible combinations of m^dim. We
    # do this by defining a mask of all possible combinations with shape
    # [m^dim x dim]. The summed up tensor then describes the weight indices
    # influencing the end vertex of each passed edge.

    # 1. Create mask.
    mask = create_mask(dim, degree)

    # 2. Apply mask.
    index = index.view(-1, m * dim)
    index = index[:, mask.view(-1)]
    index = index.view(-1, m**dim, dim)

    # 3. Sum up dimensions.
    index = index.sum(2)

    return index


def spline_weights(values, kernel_size, max_radius, degree=1):
    # Rescale values to be in range for fast calculation splines.
    values = rescale(values, kernel_size, max_radius)

    amount = weight_amount(values, degree)
    index = weight_index(values, kernel_size, degree)
    return amount, index
