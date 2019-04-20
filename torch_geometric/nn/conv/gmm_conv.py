import torch
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing

from ..inits import uniform, reset


class GMMConv(MessagePassing):
    r"""The gaussian mixture model convolutional operator from the `"Geometric
    Deep Learning on Graphs and Manifolds using Mixture Model CNNs"
    <https://arxiv.org/abs/1611.08402>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{\Theta} \cdot
        \sum_{j \in \mathcal{N}(i) \cup \{ i \}} \sum_{k=1}^K
        \mathbf{w}_k(\mathbf{e}_{i,j}) \odot \mathbf{x}_j,

    where

    .. math::
        \mathbf{w}_k(\mathbf{e}) = \exp \left( -\frac{1}{2} {\left(
        \mathbf{e} - \mathbf{\mu}_k \right)}^{\top} \Sigma_k^{-1}
        \left( \mathbf{e} - \mathbf{\mu}_k \right) \right)

    denotes a weighting function based on trainable mean vector
    :math:`\mathbf{\mu}_k` and diagonal covariance matrix
    :math:`\mathbf{\Sigma}_k`.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        dim (int): Pseudo-coordinate dimensionality.
        kernel_size (int): Number of kernels :math:`K`.
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
    """

    def __init__(self, in_channels, out_channels, dim, kernel_size, bias=True):
        super(GMMConv, self).__init__('mean')  # Use normalized gaussians.

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dim = dim
        self.kernel_size = kernel_size

        self.mu = Parameter(torch.Tensor(in_channels * kernel_size, dim))
        self.sigma = Parameter(torch.Tensor(in_channels * kernel_size, dim))
        self.lin = torch.nn.Linear(in_channels, out_channels, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        size = self.in_channels
        uniform(size, self.mu)
        uniform(size, self.sigma)
        reset(self.lin)

    def forward(self, x, edge_index, pseudo):
        """"""
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        pseudo = pseudo.unsqueeze(-1) if pseudo.dim() == 1 else pseudo

        out = self.propagate(edge_index, x=x, pseudo=pseudo)
        return self.lin(out)

    def message(self, x_j, pseudo):
        F, (E, D), K = x_j.size(1), pseudo.size(), self.mu.size(0)

        # See: https://github.com/shchur/gnn-benchmark
        gaussian = -0.5 * (pseudo.view(E, 1, D) - self.mu.view(1, K, D))**2
        gaussian = gaussian / (1e-14 + self.sigma.view(1, K, D)**2)
        gaussian = torch.exp(gaussian.sum(dim=-1))

        return x_j * gaussian.view(E, F, -1).sum(dim=-1)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)
