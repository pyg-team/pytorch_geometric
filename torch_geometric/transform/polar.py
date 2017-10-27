from math import pi as PI

import torch


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
