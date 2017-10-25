from math import pi as PI

import torch

from ..sparse import SparseTensor


def vec2ang(x, y):
    ang = torch.atan2(y, x)
    return ang + (ang <= 0).type_as(x) * 2 * PI


def polar_coordinates(vertices, edges, type=torch.FloatTensor):
    dim = vertices.size(1)

    if dim <= 1:
        raise ValueError('Invalid vertices dimensionality')

    rows, cols = edges
    starts = vertices[rows]
    ends = vertices[cols]
    v = (ends - starts).type(type)

    # Calculate Euclidean distances.
    rho = (v * v).sum(1).sqrt()

    # Append angles.
    values = [rho]
    values.extend([vec2ang(v[:, 0], v[:, i]) for i in range(1, dim)])

    return torch.stack(values, dim=1)


def euclidean_adj(vertices, edges):
    rows, cols = edges
    values = (vertices[cols] - vertices[rows])
    c = 1 / (2 * values.abs().max())
    values = c * values.float() + 0.5
    n, dim = vertices.size()
    return SparseTensor(edges, values, torch.Size([n, n, dim]))


class EuclideanAdj(object):
    def __call__(self, data):
        vertices, edges = data
        adjs = euclidean_adj(vertices, edges)
        return vertices, adjs


def mesh_adj(vertices, edges, type=torch.FloatTensor):
    n, dim = vertices.size()
    values = polar_coordinates(vertices, edges, type)
    return SparseTensor(edges, values, torch.Size([n, n, dim]))


class MeshAdj(object):
    def __call__(self, data):
        vertices, edges = data
        adjs = mesh_adj(vertices, edges)
        return vertices, adjs


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
    # NOTE: `coalesce()` doesn't work if `transpose(...)` is removed.
    adj = adj.transpose(0, 1).coalesce()
    return adj._indices()
