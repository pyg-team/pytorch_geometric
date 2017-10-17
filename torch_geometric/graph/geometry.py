from math import pi as PI

import torch


def polar_coordinates(vertices, edges):
    rows, cols = edges
    starts = vertices[rows]
    ends = vertices[cols]
    v = ends - starts

    dists = (v * v).sum(1).sqrt()

    dim = vertices.size(1)
    if dim == 2:
        return dists, vec2ang(v[:, 0], v[:, 1])
    elif dim == 3:
        return dists, vec2ang(v[:, 0], v[:, 1]), vec2ang(v[:, 0], v[:, 2])
    else:
        raise ValueError()


def vec2ang(y, x):
    ang = torch.atan2(y, x)
    return ang + (ang <= 0).type(torch.FloatTensor) * 2 * PI


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
