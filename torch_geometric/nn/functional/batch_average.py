import torch


def batch_average(input, slice):
    last_index = 0
    outputs = []
    for i in range(slice.size(0)):
        output = input[last_index:slice[i]]
        output = output.mean(dim=0)
        outputs.append(output)
        last_index = slice[i]

    return torch.stack(outputs, dim=0)
