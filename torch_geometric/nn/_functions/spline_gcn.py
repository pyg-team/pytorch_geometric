# import torch

# from ...sparse import mm, sum


def spline_gcn(adj, features, weight, bias, kernel_size, spline_degree):
    indices, dist, ang1, ang2 = adj
    # Maybe we can obtain an ang matrix with [2, |E|]?
    # TODO: Implement spline_degree.values > 1

    # COMPUTE B AND C with size [dim, spline_degree, |E|]
    # 1. Scale ang1 and ang2 to be in interval [0, 1]

    # 2. Calculate close b-spline for ang1 and ang2

    # 3. Calculate open b-spline for dist

    # 4. Concatenate to one big tensor (maybe unneccassary, will see)

    # DO THE CONVOLUTION:

    # 1.
