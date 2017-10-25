import torch


def rescale(values, kernel_size, max_radius):
    """Scale values by defining a factor for each dimension."""

    # Clamp radius (dimension 0) to its maximum allowed value.
    clamped_radius = torch.clamp(values[:, :1], max=max_radius)
    values = torch.cat([clamped_radius, values[:, 1:]], dim=1)

    kernel = torch.FloatTensor(kernel_size).type_as(values)
    # Dimension 0 with open splines has one less partition.
    kernel[0] -= 1

    boundary = [2 * PI for _ in range(len(kernel_size))]
    # Dimension 0 must be clamped to the maxium allowed radius
    boundary[0] = max_radius
    boundary = torch.FloatTensor(boundary).type_as(values)

    factor = kernel / boundary
    return factor * values
