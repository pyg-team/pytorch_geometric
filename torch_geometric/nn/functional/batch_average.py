import torch


def batch_average(input, slice):
    """Averages :obj:`input` features
    :math:`F_{in} \in \mathbb{R}^{N x M_{in}}` in the node dimension to a fixed
    batch shape :math:`[B, M_{in}]`. Batch information is given in :obj:`slice`
    containing the example slices in the node dimension.

    Example:

        >>> input = torch.FloatTensor([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])
        >>> slice = torch.LongTensor([2, 5])
        >>> output = batch_average(input, slice)
        >>> # [[1, 2], [6, 7]]
    """

    last_index = 0
    outputs = []
    for i in range(slice.size(0)):
        output = input[last_index:slice[i]]
        output = output.mean(dim=0)
        outputs.append(output)
        last_index = slice[i]

    return torch.stack(outputs, dim=0)
