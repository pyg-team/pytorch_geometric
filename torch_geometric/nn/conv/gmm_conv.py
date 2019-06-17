import torch
from torch.nn import Parameter, Linear
from torch_geometric.nn.conv import MessagePassing

from ..inits import reset, uniform, zeros

EPS = 1e-15


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
        aggr (string, optional): The aggregation scheme to use
            (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
            (default: :obj:`"add"`)
        root_weight (bool, optional): If set to :obj:`False`, the layer will
            not add transformed root node features to the output.
            (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 dim,
                 kernel_size,
                 aggr='add',
                 root_weight=True,
                 bias=True,
                 **kwargs):
        super(GMMConv, self).__init__(aggr=aggr, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dim = dim
        self.kernel_size = kernel_size

        self.lin = Linear(in_channels, out_channels * kernel_size, bias=False)
        self.mu = Parameter(torch.Tensor(kernel_size, dim))
        self.sigma = Parameter(torch.Tensor(kernel_size, dim))

        if root_weight:
            self.root = Parameter(torch.Tensor(in_channels, out_channels))
        else:
            self.register_parameter('root', None)

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.lin)
        uniform(self.kernel_size, self.mu)
        uniform(self.kernel_size, self.sigma)
        uniform(self.in_channels, self.root)
        zeros(self.bias)

    def forward(self, x, edge_index, pseudo):
        """"""
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        pseudo = pseudo.unsqueeze(-1) if pseudo.dim() == 1 else pseudo

        out = self.lin(x).view(-1, self.kernel_size, self.out_channels)
        out = self.propagate(edge_index, x=out, pseudo=pseudo)

        if self.root is not None:
            out = out + torch.matmul(x, self.root)

        if self.bias is not None:
            out = out + self.bias

        return out

    def message(self, x_j, pseudo):
        (E, D), K = pseudo.size(), self.mu.size(0)

        # See: https://github.com/shchur/gnn-benchmark
        gaussian = -0.5 * (pseudo.view(E, 1, D) - self.mu.view(1, K, D))**2
        gaussian = gaussian / (EPS + self.sigma.view(1, K, D)**2)
        gaussian = torch.exp(gaussian.sum(dim=-1, keepdim=True))  # [E, K, 1]

        return (x_j * gaussian).sum(dim=1)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)
