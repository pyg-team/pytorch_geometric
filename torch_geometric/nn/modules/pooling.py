import torch.nn.functional as F
from torch.nn import Module

from .utils.repr import repr


class GraclusMaxPool(Module):
    """Graclus Max Pooling from the `"Convolutional Neural Networks on Graphs
    with Fast Localized Spectral Filtering"
    <https://arxiv.org/abs/1606.09375>`_ paper.
    Technically applies a 1D max pooling over an input signal composed of
    several input planes with kernel_size :math:`2^{level}`.

    Args:
        level (int): Levels of cluster to pool. (default: :obj:`1`)
    """

    def __init__(self, level=1):
        super(GraclusMaxPool, self).__init__()
        self.level = level
        self.kernel_size = 2**level

    def forward(self, input):
        output = F.max_pool1d(input.t().unsqueeze(0), self.kernel_size)
        return output.squeeze(0).t()

    def __repr__(self):
        return repr(self, ['level'])
