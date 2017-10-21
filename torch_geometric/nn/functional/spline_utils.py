from math import pi as PI
from functools import reduce
import itertools

import torch


def open_spline_amount(values, degree=1):
    # Passed values must be in the range [0, num_knots -1].
    amount_grow = values.frac()
    amount_fall = 1 - amount_grow
    return torch.stack([amount_grow, amount_fall], dim=len(values.size()))


def open_spline_index(values, num_knots, degree=1):
    # Passed values must be in the range [0, num_knots - 1].
    idx_fall = values.floor().long()
    idx_fall = idx_fall - (idx_fall >= num_knots).long()
    idx_grow = 1 + idx_fall
    return torch.stack([idx_grow, idx_fall], dim=len(values.size()))


def closed_spline_amount(values, degree=1):
    # Passed values must be in the range [0, num_knots].
    amount_grow = values.frac()
    amount_fall = 1 - amount_grow
    return torch.stack([amount_grow, amount_fall], dim=len(values.size()))


def closed_spline_index(values, num_knots, degree=1):
    # Passed values must be in the range [0, num_knots].
    idx_grow = values.floor().long()
    idx_fall = idx_grow - 1
    idx_grow = idx_grow - (idx_grow >= num_knots).long() * num_knots
    idx_fall = idx_fall + (idx_fall < 0).long() * num_knots
    return torch.stack([idx_grow, idx_fall], dim=len(values.size()))


def rescale(values, kernel_size):
    kernel = torch.FloatTensor(kernel_size)
    kernel[0] -= 1  # Open spline has one less partition.

    boundary = torch.FloatTensor([2 * PI for _ in range(len(kernel_size))])
    # Radius is rescaled dependent on its max value.
    boundary[0] = values[:, 0].max()

    return (kernel / boundary) * values


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
    # Collect all indices for all dimensions.
    index = []
    for i, col in enumerate(values.t()):
        if i == 0:
            index.append(open_spline_index(col, kernel_size[i], degree))
        else:
            index.append(closed_spline_index(col, kernel_size[i], degree))

    # Stack indices to a [|E| x (degree + 1) x dim] tensor.
    index = torch.stack(index, dim=2)

    # Multiply an offset to indices in each dimension, which allows flattening
    # the indices in a later stage.
    # The offset is defined by the `kernel_size`. In the 3D case with kernel
    # [k_1, k_2, k_3]` the offset is then given by [1, k_1, k_1 * k_2].
    offset = [1]
    curr_kernel = 1
    for i in range(len(kernel_size) - 1):
        curr_kernel *= kernel_size[i]
        offset.append(curr_kernel)
    offset = torch.LongTensor(offset)

    index = offset * index

    # Finally, we can compute a flattened out version of the tensor
    # [|E| x (degree + 1) x dim] with the shape [|E| x (degree + 1)^dim], which
    # contains the summed up indices of all dimensions for all possible
    # combinations of (degree + 1)^dim.
    # The flattened out tensor then describes the weight indices influencing
    # the end vertex of each passed edge.

    print(index)

    # print(indices.size())
    # print(indices)
    return index
    # K = reduce(lambda x, y: x * y, kernel_size)
    # print('dim', dim, 'K', K)
    # print('values', values.size())
    # print(values)

    #     # Values contains polar coordinates with shape [|E| x dim].
    #     # Iterate over each coordinate and calculate spline amounts and indices.
    #     coff = 1
    #     for i, col in enumerate(values.t()):
    #         if i == 0:
    #             # Rescale values to range [0, kernel_size[i] - 1].
    #             col = (kernel_size[i] - 1) * col / col.max()

    #             amount = closed_spline_amount(col, degree)
    #             index = closed_spline_index(col, kernel_size[i], degree)
    #             print(amount)
    #         else:
    #             coff *= kernel_size[i - 1]

    #             # Rescale values to range [0, kernel_size[i]].
    #             col = kernel_size[i] * col / (2 * PI)

    #             amount = open_spline_amount(col, degree)
    #             index = open_spline_index(col, kernel_size[i], degree)
    #             print(amount)

    #         index = coff * index

    #         pass

    # print(d)

    # Calcu

    pass


def points(dim, degree):
    return itertools.product(* [range(degree) for _ in range(dim)])


def _weight_indices(spline_indices, kernel_size):
    d, m = spline_indices.size()

    # Get all possible combinations resulting in a [m^d x d] matrix.
    a = list(itertools.product(* [range(m) for _ in range(d)]))
    a = torch.LongTensor(a)

    # Add up indices column-wise, so each column contains independent indices
    # ranging globally from 0 to m*d.
    a = a + torch.arange(0, d * m, m).long()

    # Fill in the spline indices.
    a = spline_indices.view(-1)[a.view(-1)].view(m**d, d)

    if d == 3:
        p1, p2, p3 = kernel_size
        multer = [p2 * p3, p3, 1]
    elif d == 2:
        p1, p2 = kernel_size
        multer = [p2, 1]
    multer = torch.LongTensor(multer)
    a = a * multer
    a = a.sum(1)

    print(a)
    return a


def weight_amounts(spline_amounts, kernel_size):
    d, m = spline_amounts.size()

    # Get all possible combinations resulting in a [m^d x d] matrix.
    a = list(itertools.product(* [range(m) for _ in range(d)]))
    a = torch.LongTensor(a)

    a = a + torch.arange(0, d * m, m).long()
    a = spline_amounts.view(-1)[a.view(-1)].view(m**d, d)

    # TODO: Reduce multiply
    a = a.mul(1)

    print(a)
