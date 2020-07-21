import torch
from torch import Tensor
import torch.nn.functional as F


class PairNorm(torch.nn.Module):
    r"""Applies pair normalization over the inputs as described
    in the `"PairNorm: Tackling Oversmoothing in GNNs"
    <https://arxiv.org/abs/1909.12223>`_ paper

    .. math::
        \mathbf{\widetilde{x}}_i^c &= \mathbf{\widetilde{x}}_i - \frac{1}{n}
        \sum_{i=1}^{n} \mathbf{\widetilde{x}}_i \\
        \mathbf{\dot{x}}_i &= s\sqrt{n} . \frac{\mathbf{\widetilde{x}}_i^c}
        {\sqrt{\| \mathbf{\widetilde{X}}^c} \|_F^2}

    Args:
        mode (str, optional): Mode of normalization
            (default: :obj:`PN`)
        scale (int, optional): Scaling factor of normalization
            (default, :obj:`1`)
    """
    def __init__(self, mode: str = 'PN', scale: int = 1):
        super(PairNorm, self).__init__()

        assert mode in ['None', 'PN', 'PN-SI', 'PN-SCS']

        self.mode = mode
        self.scale = scale

    def forward(self, x: Tensor):
        if self.mode == 'None':
            return x

        column_mean = x.mean(dim=0)
        if self.mode == 'PN':
            x = x - column_mean
            row_norm_mean = (1e-6 + x.pow(2).sum(dim=1).mean()).sqrt()
            x = self.scale * x / row_norm_mean

        elif self.mode == 'PN-SI':
            x = x - column_mean
            row_norm_individual = (
                1e-6 + x.pow(2).sum(dim=1, keepdim=True)
            ).sqrt()
            x = self.scale * x / row_norm_individual

        elif self.mode == 'PN-SCS':
            row_norm_individual = (
                1e-6 + x.pow(2).sum(dim=1, keepdim=True)
            ).sqrt()
            x = self.scale * x / row_norm_individual - column_mean

        return x

    def __repr__(self):
        return ('{}(mode={})').format(self.__class__.__name__, self.mode)
