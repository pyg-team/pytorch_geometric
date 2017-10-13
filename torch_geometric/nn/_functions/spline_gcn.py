import torch

from ...sparse import mm, sum


def spline_gcn(adj, features, weight, bias, kernel_size, spline_degree):
    indices, dist, ang1, ang2 = adj
    # TODO: Implement spline_degree.values > 1

    # Compute B and C:
    # Than B and C will have size [dim, degree[i] + 1, |E|]

    # We get

    # So wie geht es?
