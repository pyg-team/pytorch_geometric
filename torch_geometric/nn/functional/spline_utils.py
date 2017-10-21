from math import pi as PI
import itertools

import torch


def open_spline_amount(values, degree=1):
    # Passed values must be in the range [0, num_knots -1].
    amount_grow = values.frac()
    amount_fall = 1 - amount_grow
    return torch.stack([amount_grow, amount_fall], dim=len(values.size()))


def open_spline_index(values, kernel_size, degree=1):
    # Passed values must be in the range [0, num_knots - 1].
    if isinstance(kernel_size, (list, tuple)):
        kernel_size = torch.LongTensor(kernel_size)

    idx_fall = values.floor().long()
    idx_fall = idx_fall - (idx_fall >= kernel_size).long()
    idx_grow = 1 + idx_fall
    return torch.stack([idx_grow, idx_fall], dim=len(values.size()))


def closed_spline_amount(values, degree=1):
    # Passed values must be in the range [0, num_knots].
    amount_grow = values.frac()
    amount_fall = 1 - amount_grow
    return torch.stack([amount_grow, amount_fall], dim=len(values.size()))


def closed_spline_index(values, kernel_size, degree=1):
    # Passed values must be in the range [0, num_knots].
    if isinstance(kernel_size, (list, tuple)):
        kernel_size = torch.LongTensor(kernel_size)

    idx_grow = values.floor().long()
    idx_fall = idx_grow - 1
    idx_grow = idx_grow - (idx_grow >= kernel_size).long() * kernel_size
    idx_fall = idx_fall + (idx_fall < 0).long() * kernel_size
    return torch.stack([idx_grow, idx_fall], dim=len(values.size()))


def rescale(values, kernel_size):
    # Scale values by defining a facot for each dimension.

    kernel = torch.FloatTensor(kernel_size)
    # Dimension 0 with open splines has one less partition.
    kernel[0] -= 1

    # TODO: this doesn't work batch-wise.
    boundary = torch.FloatTensor([2 * PI for _ in range(len(kernel_size))])
    # Dimension 0 is bounded dependent on its max value.
    boundary[0] = values[:, 0].max()

    factor = kernel / boundary
    return factor * values


def create_mask(dim, degree):
    m = degree + 1
    mask = list(itertools.product(* [range(m) for _ in range(dim)]))
    mask = torch.LongTensor(mask)
    mask += torch.arange(0, dim * m, m).long()
    return mask


def spline_weights(values, kernel_size, degree=1):
    # Rescale values to be in range for fast calculation splines.
    values = rescale(values, kernel_size)

    amount = weight_amount(values, kernel_size, degree)
    index = weight_index(values, kernel_size, degree)
    return amount, index


def weight_amount(values, kernel_size, degree=1):
    # values = num_knots * values / (2 * PI)
    pass


def weight_index(values, kernel_size, degree=1):
    dim = len(kernel_size)
    m = degree + 1

    # Collect all spline indices for all dimensions with final shape
    # [|E| x dim x m]. Dimension 0 needs to be computed by using open splines,
    # whereas all other dimensions should be handled by closed splines.

    index_0 = open_spline_index(values[:, :1], kernel_size[:1], degree)
    index_remain = closed_spline_index(values[:, 1:], kernel_size[1:], degree)
    index = torch.cat([index_0, index_remain], dim=1)

    # Broadcast element-wise multiply an offset to indices in each dimension,
    # which allows flattening the indices in a later stage.
    # The offset is defined by the `kernel_size`, e.g. in the 3D case with
    # kernel [k_1, k_2, k_3]` the offset is given by [1, k_1, k_1 * k_2].

    # 1. Calculate offset.
    # TODO: more elegant, please!
    offset = [1]
    curr_kernel = 1
    for i in range(len(kernel_size) - 1):
        curr_kernel *= kernel_size[i]
        offset.append(curr_kernel)
    offset = torch.LongTensor([offset]).t()

    # 2. Apply offset.
    index = offset * index

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
