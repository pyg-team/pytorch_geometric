from math import pi as PI
import itertools

import torch


def open_spline(values, num_knots, degree=1):
    values = num_knots * values / values.max()

    b_1 = values.frac()
    b_2 = 1 - b_1
    B = torch.stack([b_1, b_2], dim=len(b_1.size()))

    c_2 = values.floor().long()
    c_2 = c_2 - (c_2 >= num_knots).long()
    c_1 = 1 + c_2
    C = torch.stack([c_1, c_2], dim=len(c_1.size()))

    return B, C


def closed_spline(values, num_knots, degree=1):
    values = num_knots * values / (2 * PI)

    b_1 = values.frac()
    b_2 = 1 - b_1
    B = torch.stack([b_1, b_2], dim=len(b_1.size()))

    c_1 = values.floor().long()
    c_2 = c_1 - 1
    c_1 = c_1 - (c_1 >= num_knots).long() * num_knots
    c_2 = c_2 + (c_2 < 0).long() * num_knots
    C = torch.stack([c_1, c_2], dim=len(c_1.size()))

    return B, C


def points(dim, degree):
    return itertools.product(* [range(degree) for _ in range(dim)])


def weight_indices(spline_indices, kernel_size):
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
