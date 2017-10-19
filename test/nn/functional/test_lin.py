from torch.nn.functional import conv1d


def lin(features, weight, bias):
    weight = weight.t()
    outs, ins = weight.size()
    weight = weight.view(outs, ins, 1)

    return conv1d(features, weight, bias)
