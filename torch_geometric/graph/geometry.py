from math import pi as PI

import torch


def polar_coordinates(adj_indices, points):
    rows, cols = adj_indices
    starts = points[rows]
    ends = points[cols]
    directions = ends - starts

    dists = directions * directions
    dists = dists.sum(1).sqrt()

    vec_y = directions[:, 0]  # Save for later usage.
    ang1 = vec2ang(vec_y, directions[:, 1])

    dim = points.size()[1]
    if dim == 2:
        return dists, ang1
    elif dim == 3:
        ang2 = vec2ang(vec_y, directions[:, 2])
        return dists, ang1, ang2
    else:
        return ValueError()


def vec2ang(y, x):
    ang = torch.atan2(y, x)
    neg = (ang <= 0).type(torch.FloatTensor)

    if torch.cuda.is_available():
        neg = neg.cuda()

    return ang + (neg * 2 * PI)
