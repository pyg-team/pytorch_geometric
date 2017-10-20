from functools import reduce
from math import pi as PI
import itertools

import torch

# from torch import nn

from ...sparse import sum


def spline_gcn(adj, features, weight, spline_degree=1, bias=None):
    indices = adj._indices()
    _, cols = indices

    # Convert to [|E| x M_in] feature matrix and calculate [|E| x M_out].
    output = features[cols]
    output = transform(adj, output, weight, spline_degree)

    # Convolution via sparse row sum. Converts [|E| x M_out] feature matrix to
    # [n x M_out] feature matrix.
    adj = torch.sparse.FloatTensor(indices, output, adj.size())
    output = sum(adj, dim=1)

    if bias is not None:
        output += bias

    return output


def transform(adj, features, weight, spline_degree=1):
    values = adj._values()
    rows, cols = adj._indices()

    M_in, M_out = weight.size()[0:2]
    kernel_size = weight.size()[2:]
    dim = len(kernel_size)
    m = spline_degree + 1
    # print('M_in', M_in, 'M_out', M_out, 'kernel_size', kernel_size)
    # print('dim', dim, 'm', m)

    # Compute B and C, resulting in two [dim x m x |E|] tensors.
    # B saves the values of the spline functions with non-zero values.
    # The values in C point to the corresponding spline function indices.
    # B1, C1 = open_spline(values[:, 0], partitions=kernel_size[0] - 1)
    # B, C = [B1], [C1]
    # for i in range(1, dim):
    #     Bi, Ci = closed_spline(values[:, i], partitions=kernel_size[i])
    #     B.append(Bi)
    #     C.append(Ci)
    # B = torch.stack(B, dim=0)
    # C = torch.stack(C, dim=0)
    # print('B', B.size(), 'C', C.size())

    # for p in points(dim, m):
    #     c = C[p]
    #     print('c', c.size())
    #     # print(c)
    #     b = B[p]
    #     print('weight normal', weight.size())
    #     for i in range(M_in):
    #         w = weight[i]
    #         print('w', w.size())
    #         w = w[:, c]
    #         # w = w[:, p]

    #         print('w', w.size())
    #         C[p]
    #         print(i)

    # pass


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


def weight_indices(spline_indices, kernel_size):
    P = reduce(lambda x, y: x * y, kernel_size)
    d, m = spline_indices.size()
    print('kernel_size', kernel_size)
    print('P', P, 'd', d, 'm', m)

    # print(spline_indices)
    c = spline_indices

    print(c)
    c = c.view(-1)
    # print(c)

    a = list(itertools.product(*[range(m) for _ in range(d)]))
    a = torch.LongTensor(a)

    adder = torch.LongTensor([0, 2, 4])

    a = a + adder
    a = a.view(-1)
    a = c[a]
    a = a.view(8, 3)
    print(a)
    # a = a.view(-1)
    # print(a)
    # print(a[c])

    p1, p2, p3 = kernel_size
    multer = [p2 * p3, p3, 1]
    multer = torch.LongTensor(multer)
    a = a * multer
    a = a.sum(1)
    print(a)
    return
    print(multer.size(), ' * ', spline_indices.size())
    c = spline_indices.t() * multer
    c = c.t()

    # # print(c.chunk(d, 0))
    # # c = c.chunk(d, 0)
    print(c)

    return
    a = a.t()
    b = 1 - a
    # a = a.t()
    # b = 1 - a

    # # print(a)
    # c = torch.cat([a, b], dim=0)
    # d = 1 - c
    # e = torch.cat([c, d], dim=1)

    print(a)
    # print(b)

    return

    for i in itertools.combinations_with_():
        print(i)
        # print(c[i, 0])
        # print(i)

    return
    # print(i)
    # return
    # print(i)

    # print(c)

    # return
    # print(p1, p2)
    # print('d x m', spline_indices[0])
    # print(spline_indices[1])

    multer = torch.LongTensor([[p2, 1]])

    # multer = torch.LongTensor([3, 1])
    # adder = torch.LongTensor([4, 0])

    print(spline_indices[0].size())
    print(multer.size())
    c = torch.mm(multer, spline_indices[0])
    print('c', c)

    pass
