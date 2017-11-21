import torch


def batch_average(input, slice):
    """Averages ``input`` features in the node dimension. Batch information is
    given by ``slice``.

    Example:

        >>> input = torch.FloatTensor([[1, 2], [3, 4], [5, 6], [7, 8]])
        >>> slice = torch.LongTensor([2, 4])
        >>> output = batch_average(input, slice)
        >>> # [[2, 3], [6, 7]]
    """

    last_index = 0
    outputs = []
    for i in range(slice.size(0)):
        output = input[last_index:slice[i]]
        output = output.mean(dim=0)
        outputs.append(output)
        last_index = slice[i]

    return torch.stack(outputs, dim=0)
