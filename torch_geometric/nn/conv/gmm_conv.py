import torch
from torch.nn import Parameter
from torch_scatter import scatter_add

from ..inits import uniform, reset


class GMMConv(torch.nn.Module):
    r"""The gaussian mixture model convolutional operator from the `"Geometric
    Deep Learning on Graphs and Manifolds using Mixture Model CNNs"
    <https://arxiv.org/abs/1611.08402>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{\Theta} \cdot
        \sum_{j \in \mathcal{N}(i) \cup \{ i \}} w(\mathbf{e}_{i,j}) \odot
        \mathbf{x}_j,

    where

    .. math::
        w_m(\mathbf{e}) = \exp \left( -\frac{1}{2} {\left( \mathbf{e} -
        \mathbf{\mu}_m \right)}^{\top} \Sigma_m^{-1} \left( \mathbf{e} -
        \mathbf{\mu}_m \right) \right)

    with trainable mean vector :math:`\mathbf{\mu}_m` and diagonal covariance
    matrix :math:`\mathbf{\Sigma}_m` for each input channel :math:`m`.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        dim (int): Pseudo-coordinate dimensionality.
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
    """

    def __init__(self, in_channels, out_channels, dim, bias=True):
        super(GMMConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dim = dim

        self.mu = Parameter(torch.Tensor(in_channels, dim))
        self.sigma = Parameter(torch.Tensor(in_channels, dim))
        self.lin = torch.nn.Linear(in_channels, out_channels, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        size = self.in_channels
        uniform(size, self.mu)
        uniform(size, self.sigma)
        reset(self.lin)

    def forward(self, x, edge_index, pseudo):
        """"""
        # See https://github.com/shchur/gnn-benchmark for the reference
        # TensorFlow implementation.
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        pseudo = pseudo.unsqueeze(-1) if pseudo.dim() == 1 else pseudo
        row, col = edge_index

        F, (E, D) = x.size(1), pseudo.size()

        gaussian = -0.5 * (pseudo.view(E, 1, D) - self.mu.view(1, F, D))**2
        gaussian = gaussian / (1e-14 + self.sigma.view(1, F, D)**2)
        gaussian = torch.exp(gaussian.sum(dim=-1))

        out = scatter_add(x[col] * gaussian, row, dim=0, dim_size=x.size(0))
        out = self.lin(out)

        return out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)
