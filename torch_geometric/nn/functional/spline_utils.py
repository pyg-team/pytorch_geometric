from math import pi as PI
import itertools

import torch


def open_spline(values, num_knots, degree=1):
    dim = len(values.size())
    values = num_knots * values / values.max()

    amount_grow = values.frac()
    amount_fall = 1 - amount_grow
    amount = torch.stack([amount_grow, amount_fall], dim)

    idx_fall = values.floor().long()
    idx_fall = idx_fall - (idx_fall >= num_knots).long()
    idx_grow = 1 + idx_fall
    idx = torch.stack([idx_grow, idx_fall], dim)

    return amount, idx


def closed_spline(values, num_knots, degree=1):
    dim = len(values.size())
    values = num_knots * values / (2 * PI)

    amount_grow = values.frac()
    amount_fall = 1 - amount_grow
    amount = torch.stack([amount_grow, amount_fall], dim)

    idx_grow = values.floor().long()
    idx_fall = idx_grow - 1
    idx_grow = idx_grow - (idx_grow >= num_knots).long() * num_knots
    idx_fall = idx_fall + (idx_fall < 0).long() * num_knots
    idx = torch.stack([idx_grow, idx_fall], dim)

    return amount, idx


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
