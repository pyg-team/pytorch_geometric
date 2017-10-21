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
