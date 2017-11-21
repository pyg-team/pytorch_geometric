import torch.nn.functional as F
from torch.nn import Module


class GraclusMaxPool(Module):
    """Graclus Max Pooling from the `"Convolutional Neural Networks on Graphs
    with Fast Localized Spectral Filtering"
    <https://arxiv.org/abs/1606.09375>`_ paper.
    Technically applies a 1D max pooling over an input signal composed of
    several input planes with kernel_size :math:`2^{level}`.

    Args:
        in_features (int): Size of each input sample.
        out_features (int): Size of each output sample.
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
    """

    def __init__(self, level):
        super(GraclusMaxPool, self).__init__()
        self.level = level
        self.kernel_size = 2**level

    def forward(self, input):
        output = F.max_pool1d(input.t().unsqueeze(0), self.kernel_size)
        return output.squeeze(0).t()

    def __repr__(self):
        s = '{name}(level={level})'
        return s.format(name=self.__class__.__name__, **self.__dict__)
