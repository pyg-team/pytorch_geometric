import time
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


def edges_from_faces(faces):
    # Get directed edges.
    edges = torch.cat((faces[:, 0:2], faces[:, 1:3], faces[:, 0:3:2]))
    n = faces.max() + 1

    # Build directed adjacency matrix.
    adj = torch.sparse.FloatTensor(edges.t(),
                                   torch.ones(edges.size(0)),
                                   torch.Size([n, n]))

    # Convert to undirected adjacency matrix.
    adj = adj + adj.t()

    # Remove duplicate indices.
    # NOTE: This doesn't work if transpose(...) is removed.
    adj = adj.transpose(0, 1).coalesce()
    return adj._indices()
