from torch.nn.functional import conv1d


def lin(features, weight, bias=None):
    features = features.t().unsqueeze(0)
    return conv1d(features, weight, bias).squeeze().t()
