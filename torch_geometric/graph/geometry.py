from math import pi as PI

import torch


def vec2ang(x, y):
    ang = torch.atan2(x, y)
    return ang + (ang <= 0).type(torch.FloatTensor) * 2 * PI


def polar_coordinates(vertices, edges):
    rows, cols = edges
    starts = vertices[rows]
    ends = vertices[cols]
    v = ends - starts

    # Calculate Euclidean distances.
    rho = (v * v).sum(1).sqrt()

    # Dimensionality case differentation.
    dim = vertices.size(1)
    if dim == 2:
        return rho, vec2ang(v[:, 1], v[:, 0])
    elif dim == 3:
        return rho, vec2ang(v[:, 1], v[:, 0]), vec2ang(v[:, 2], v[:, 0])

    raise ValueError()


def edges_from_faces(faces):
    """Get undirected edges from triangular faces."""

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
