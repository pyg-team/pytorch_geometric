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
    n = faces.max() + 1
    dim = faces.size(1)

    if dim == 2:
        edges = faces
    if dim == 3:
        edges = torch.cat((faces[:, 0:2], faces[:, 0:3:2]))
    else:
        return ValueError()

    # # edges = edges:index(2, torch.linspace(2,1)):long()
    # # edges = torch.cat((edges, edges[:, ::-1]))
    # inv_idx = torch.arange(dim - 1, -1, -1).long()
    # # edges = edges[:, inv_idx]
    # print(inv_idx)
    # # edges.index_select(0, torch.arange(edges.size(0) - 1, -1).long())
    # print(edges.size())

    # print(n)

    # Sort the adjacencies row-wise.
    edges = edges.t()
    sorted, indices = torch.sort(edges[0], 0)
    edges = torch.cat((sorted, edges[1][indices])).view(2, -1)
    return edges
