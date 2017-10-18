from math import pi as PI

import torch


def vec2ang(x, y):
    ang = torch.atan2(y, x)
    return ang + (ang <= 0).type(torch.FloatTensor) * 2 * PI


def polar_coordinates(vertices, edges):
    dim = vertices.size(1)
    assert dim > 1, 'Invalid vertices dimensionality.'

    rows, cols = edges
    starts = vertices[rows]
    ends = vertices[cols]
    v = ends - starts

    # Calculate Euclidean distances.
    rho = (v * v).sum(1).sqrt()

    # Append angles.
    values = [rho]
    values.extend([vec2ang(v[:, 0], v[:, i]) for i in range(1, dim)])

    return torch.stack(values, dim=1)


def mesh_adj(vertices, edges):
    n, dim = vertices.size()
    return torch.sparse.FloatTensor(edges,
                                    polar_coordinates(vertices, edges),
                                    torch.Size([n, n, dim]))


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
