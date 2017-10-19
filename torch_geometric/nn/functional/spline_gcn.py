from math import pi as PI

import torch
from torch import nn

# from ...sparse import mm, sum


def spline_gcn(adj, features, weight, kernel_size, spline_degree=1, bias=None):
    values = adj._values()
    rows, cols = adj._indices()

    # ---- PREPROCESSING ------------------------------------------------------

    # Compute B and C, resulting in two [|E| x dim x (degree + 1)] tensors.
    # B saves the values of the spline functions with non-zero values.
    # The values in C point to the corresponding spline function indices.
    B1, C1 = open_spline(values[:, 0], partitions=kernel_size[0])
    B, C = [B1], [C1]
    for dim in range(1, len(kernel_size)):
        Bi, Ci = closed_spline(values[:, dim], partitions=kernel_size[dim])
        B.append(Bi)
        C.append(Ci)

    B = torch.stack(B, dim=0).transpose(1, 2).transpose(0, 1)
    C = torch.stack(C, dim=0).transpose(1, 2).transpose(0, 1)
    B = B.contiguous().view(-1, 4)
    C = C.contiguous().view(-1, 4)
    print(B.size(), C.size())

    # ---- CONVOLUTION - ------------------------------------------------------

    print('conv')
    # 1. We need to compute [|E|, M_out]:
    row_features = features[rows]
    col_features = features[cols]
    print(col_features.size())
    print(B.size())
    # print(row_features.size())
    # print(col_features.size())

    # col_features = col_features * B  # results to
    c = torch.bmm(col_features.view(8, 2, 1), B.view(8, 1, 4))
    # c => |E| x M_in x (d+k+1)
    print(c.size())
    print(c.view(8, -1))

    # print(B[0])
    # print(col_features[0])
    # print((col_features * B)[0])

    # We are only interested in the col_features!!!
    # This needs to be converted from [|E|, M_in] to [|E|, M_out].

    # bla = nn.parallel.scatter(col_features, [0])

    # print(bla)

    # col_features *

    # Get b => [|E|, (d * k + 1)]
    # Get c => [|E|, (d * k + 1)]
    # for each edge:
    # Compute \sum_{b_ij} b_{ij} * f (element-wise) * W_{C_{ij}

    # D

    # -------------------------------------------------------------------------
    # DO THE CONVOLUTION:
    # C = B.view(-1, dim * (spline_degree + 1))
    # C = C.view(-1, dim * (spline_degree + 1))

    # 1. We need to compute [|E|, M_out]:
    #   1. Slice the features F => [|E|, M_in]
    #
    #   2. for one i in |E| compute the following:
    #       1. Get b := B_{i} => [dim, spline_degree + 1] (slice, no problem)
    #       2. Get c := C_{i} => [dim, spline_degree + 1] (slice, no problem)
    #       3. Get w := W_{c} => [dim, spline_degree + 1, M_in, M_out]
    #               (slice, no problem)
    #       4. Get the current row r := R_{i} (no problem)
    #       5. Get f := F_{i} => [M_in]
    #
    #       6. Compute d := B * W => [dim, spline_degree + 1, M_in, M_out]
    #       7. Compute e := [dim, spline_degree + 1, M_out]
    #       8. Sum to [M_out]
    #
    # TODO: it's maybe better to flatten [dim, spline_degree + 1 to one
    # dimension.

    # 2. Then we can do a `sum` over the rows to obtain [N, M_out]
    # 3. Finally we can weighten the root nodes (extra weights?)

    # -------------------------------------------------------------------------
    # TODO: Implement spline_degree.values > 1
    pass


def closed_spline(values, partitions, degree=1):
    values = partitions * values / (2 * PI)

    b_1 = values.frac()
    b_2 = 1 - b_1
    B = torch.stack([b_1, b_2], dim=0)

    c_1 = values.floor().long()
    c_2 = c_1 - 1
    c_1 = c_1 - (c_1 >= partitions).type(torch.LongTensor) * partitions
    c_2 = c_2 + (c_2 < 0).type(torch.LongTensor) * partitions
    C = torch.stack([c_1, c_2], dim=0)

    return B, C


def open_spline(values, partitions, degree=1):
    values = partitions * values / values.max()

    b_1 = values.frac()
    b_2 = 1 - b_1
    B = torch.stack([b_1, b_2], dim=0)

    c_2 = values.floor().long()
    c_2 = c_2 - (c_2 >= partitions).type(torch.LongTensor)
    c_1 = 1 + c_2
    C = torch.stack([c_1, c_2], dim=0)

    return B, C
