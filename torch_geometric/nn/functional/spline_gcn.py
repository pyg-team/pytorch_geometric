from functools import reduce
from math import pi as PI
import itertools

import torch

# from torch import nn

# from ...sparse import mm, sum

# def spline_gcn(adj, features, weight, kernel_size, spline_degree=1, bias=None):
#     rows, cols = adj._indices()
#     values = adj._values()

#     # ---- PREPROCESSING ------------------------------------------------------

#     # Compute B and C, resulting in two [dim x (degree + 1) x |E|] tensors.
#     # B saves the values of the spline functions with non-zero values.
#     # The values in C point to the corresponding spline function indices.
#     B1, C1 = open_spline(values[:, 0], partitions=kernel_size[0])
#     B, C = [B1], [C1]
#     for dim in range(1, len(kernel_size)):
#         Bi, Ci = closed_spline(values[:, dim], partitions=kernel_size[dim])
#         B.append(Bi)
#         C.append(Ci)
#     B = torch.stack(B, dim=0)
#     C = torch.stack(C, dim=0)
#     print('B:', B.size(), '=> d x m x |E|')
#     print('C:', C.size(), '=> d x m x |E|')
#     return
#     print(C[1])

#     # ---- CONVOLUTION - ------------------------------------------------------

#     M_in = weight.size(0)
#     M_out = weight.size(1)
#     m = spline_degree + 1
#     d = values.size(-1)
#     k = m**d
#     print('kernel: ', kernel_size)
#     print('M_in: ', M_in)
#     print('M_out: ', M_out)
#     print('m: ', m)
#     print('d: ', d)
#     # print('k: ', k)
#     print('weight: ', weight.size())
#     E = values.size(0)
#     print('Num Edges: ', E)

#     # print(C)
#     # a = C[[0, 1], [1, 0]]
#     # print('a', a.size())
#     # print(a)

#     # y = torch.arange(0, d)

#     # for p in itertools.product(*[torch.arange(0, m).long() for _ in range(d)]):

#     C = C.view(m * d, -1)
#     B = B.view(m * d, -1)
#     print('new C size: ', C.size())

#     for k, p in enumerate(itertools.product(*[range(m) for _ in range(d)])):
#         print('Point: ', p)
#         kkk = torch.arange(0, d).long() * m
#         p = torch.LongTensor(p) + kkk
#         p = p.repeat(E).view(-1, d).t()
#         print('Index: ', k)

#         c = torch.gather(C, 0, p).t()
#         b = torch.gather(B, 0, p).t()

#         print(c)
#         # c = c + kkk
#         # c = c.sum(1)
#         return

#         for i in range(M_in):
#             print('index M_in: ', i)
#             w = weight[i, :]
#             print(w.size())
#             print(c)
#             w = weight[:, c]

#             # torch.gather(C, 1, p)

#             # Zu diesem P muessen nun die Indices aus C indiziert werden.
#             # Gut, wie geht das?

#             return
#             # for p in itertools.product(*[range(m) for _ in range(d)]):
#             #     p = torch.LongTensor(p) + torch.arange(0, d).long() * m
#             #     print('Point: ', p)

#             #     c = C[p]
#             #     w = weight[i]

#             #     w = w.view(1, -1)

#             #     print(c)

#             #     print(c.size())
#             #     print(w.size())

#             #     return

#             #     w = weight[i]
#             #     print(w.size())
#             #     torch.gather(w, 2, c)
#             #     return
#             # w = weight[]
#             # print(weight.size())

#             # w = weight[i, :, c]
#             # print(c)
#             pass
#     # C[]

#     # Get the indices of C

#     # print(p)

#     # pass
#     # w_gath = weight[]

#     # print(p)

#     # print(list(zip([0, 1], [0, 1])))

#     # print('conv')
#     # 1. We need to compute [|E|, M_out]:
#     # row_features = features[rows]
#     # col_features = features[cols]
#     # print(col_features.size())
#     # print(B.size())
#     # print(row_features.size())
#     # print(col_features.size())

#     # col_features = col_features * B  # results to
#     # c = torch.bmm(col_features.view(8, 2, 1), B.view(8, 1, 4))
#     # c => |E| x M_in x (d+k+1)
#     # print(c.size())
#     # print(c.view(8, -1))

#     # print(B[0])
#     # print(col_features[0])
#     # print((col_features * B)[0])

#     # We are only interested in the col_features!!!
#     # This needs to be converted from [|E|, M_in] to [|E|, M_out].

#     # bla = nn.parallel.scatter(col_features, [0])

#     # print(bla)

#     # col_features *

#     # Get b => [|E|, (d * k + 1)]
#     # Get c => [|E|, (d * k + 1)]
#     # for each edge:
#     # Compute \sum_{b_ij} b_{ij} * f (element-wise) * W_{C_{ij}

#     # D

#     # -------------------------------------------------------------------------
#     # DO THE CONVOLUTION:
#     # C = B.view(-1, dim * (spline_degree + 1))
#     # C = C.view(-1, dim * (spline_degree + 1))

#     # 1. We need to compute [|E|, M_out]:
#     #   1. Slice the features F => [|E|, M_in]
#     #
#     #   2. for one i in |E| compute the following:
#     #       1. Get b := B_{i} => [dim, spline_degree + 1] (slice, no problem)
#     #       2. Get c := C_{i} => [dim, spline_degree + 1] (slice, no problem)
#     #       3. Get w := W_{c} => [dim, spline_degree + 1, M_in, M_out]
#     #               (slice, no problem)
#     #       4. Get the current row r := R_{i} (no problem)
#     #       5. Get f := F_{i} => [M_in]
#     #
#     #       6. Compute d := B * W => [dim, spline_degree + 1, M_in, M_out]
#     #       7. Compute e := [dim, spline_degree + 1, M_out]
#     #       8. Sum to [M_out]
#     #
#     # TODO: it's maybe better to flatten [dim, spline_degree + 1 to one
#     # dimension.

#     # 2. Then we can do a `sum` over the rows to obtain [N, M_out]
#     # 3. Finally we can weighten the root nodes (extra weights?)

#     # -------------------------------------------------------------------------
#     # TODO: Implement spline_degree.values > 1
#     pass


def spline_gcn(adj, features, weight, kernel_size, spline_degree=1, bias=None):
    indices = adj._indices()
    _, cols = indices
    output = features[cols]
    output = transform(adj, output, kernel_size, spline_degree)

    adj = torch.sparse.FloatTensor(indices, output, adj.size())
    output = sum(adj, dim=1)

    if bias is not None:
        output += bias

    return output


def transform(adj, features, kernel_size, spline_degree=1):
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


def points(dim, degree):
    return itertools.product(*[range(degree) for _ in range(dim)])


# def weight_indices(spline_indices, kernel_size):
#     P = reduce(lambda x, y: x * y, kernel_size)
#     d, m = spline_indices.size()
#     return
#     print(P, d, m)

#     p1, p2, p3 = kernel_size
#     multer = [p2 * p3, p3, 1]
#     multer = torch.LongTensor(multer)
#     print(multer.size(), ' * ', spline_indices.size())
#     c = spline_indices.t() * multer
#     c = c.t()

#     # print(c.chunk(d, 0))
#     # c = c.chunk(d, 0)
#     print(c)

#     for i in itertools.product(*[[0, 1], [0, 1], [0, 1]]):
#         print(i)
#         print(c[i, 0])
#         # print(i)

#     return
#     # print(i)
#     # return
#     # print(i)

#     # print(c)

#     # return
#     # print(p1, p2)
#     # print('d x m', spline_indices[0])
#     # print(spline_indices[1])

#     multer = torch.LongTensor([[p2, 1]])

#     # multer = torch.LongTensor([3, 1])
#     # adder = torch.LongTensor([4, 0])

#     print(spline_indices[0].size())
#     print(multer.size())
#     c = torch.mm(multer, spline_indices[0])
#     print('c', c)

#     pass

# def bla(values, kernel_size):
#     B1, C1 = open_spline(values[:, 0], partitions=kernel_size[0])
#     B, C = [B1], [C1]
#     for dim in range(1, len(kernel_size)):
#         Bi, Ci = closed_spline(values[:, dim], partitions=kernel_size[dim])
#         B.append(Bi)
#         C.append(Ci)
#     B = torch.stack(B, dim=0)
#     C = torch.stack(C, dim=0)
