from torch.nn.functional import conv1d


def lin(features, weight, bias=None):
    features = features.view(1, features.size(0), weight.size(1))
    features = features.transpose(2, 1)

    return conv1d(features, weight, bias).squeeze().t()
