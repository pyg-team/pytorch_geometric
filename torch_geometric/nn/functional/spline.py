import torch


def spline(values, kernel_size, is_open_spline, degree):
    """
    Args:
        values (Tensor): 1d or 2d tensor.
        kernel_size (list)
        is_open_spline (list)
        spline_degree (int, optional): B-Spline degree. (default: 1)
    """

    if degree != 1:
        raise NotImplementedError()

    is_closed_spline = 1 - is_open_spline
    kernel_size = kernel_size - is_open_spline

    values = values.unsqueeze(1) if len(values.size()) < 2 else values
    values *= kernel_size.type_as(values)

    amount = values.frac()
    amount = torch.stack([amount, 1 - amount], dim=len(values.size()))

    index_g = values.floor().long()
    index_g -= is_open_spline * (index_g >= kernel_size).type_as(index_g)
    index_f = index_g + is_open_spline

    # index -= is_closed_spline * (
    #     index >= kernel_size).type_as(index) * kernel_size
    index_g -= is_closed_spline
    index_g += is_closed_spline * (index_g < 0).type_as(index_g) * kernel_size
    index_f -= is_closed_spline * (
        index_f >= kernel_size).type_as(index_f) * kernel_size

    index = torch.stack([index_f, index_g], dim=len(values.size()))

    return amount, index
