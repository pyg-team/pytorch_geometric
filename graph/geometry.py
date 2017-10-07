import torch


def polar_coordinates(adj_indices, points):
    starts = points[adj_indices[0]]
    ends = points[adj_indices[1]]
    directions = ends - starts

    dists = directions * directions
    dists = dists.sum(1)

    angle1 = torch.atan2(directions[:, 0], directions[:, 1])
    # TODO: 0 => 2pi
    # TODO: No negative indices

    # TODO: second angle
    return dists, angle1


i = torch.LongTensor([
    [0, 0, 1, 1],
    [1, 2, 0, 2],
])
points = torch.FloatTensor([[0, 0], [3, 4], [0, 2]])

a = polar_coordinates(i, points)
print(a)
