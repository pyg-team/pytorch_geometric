import torch


def stack(sequence, horizontal=True, vertical=True):
    if not horizontal and not vertical:
        raise ValueError('stack expects a direction')

    if len(sequence) == 0:
        raise ValueError('stack expects a non-empty sequence of sparse '
                         'tensors')

    y_sum, x_sum = (0, 0)
    indices = []
    slices = []
    for mat in sequence:
        # Offset sparse matrix indices.
        offset = [[y_sum if vertical else 0], [x_sum if horizontal else 0]]
        offset = torch.LongTensor(offset)
        indices.append(mat._indices() + offset)

        # Calculate new offset for next iteration.
        y, x = (mat.size(0), mat.size(1))
        y_sum = y_sum + y if vertical else max(y_sum, y)
        x_sum = x_sum + x if horizontal else max(x_sum, x)

        if vertical and horizontal:
            slices.append([y_sum, x_sum])
        elif vertical:
            slices.append(y_sum)
        elif horizontal:
            slices.append(x_sum)

    # Concat all indices and values to one new large sparse matrix.
    indices = torch.cat(indices, dim=1)
    values = torch.cat([mat._values() for mat in sequence])
    size = torch.Size([y_sum, x_sum, *list(sequence[0].size()[2:])])
    slices = torch.LongTensor(slices)

    return torch.sparse.FloatTensor(indices, values, size), slices
