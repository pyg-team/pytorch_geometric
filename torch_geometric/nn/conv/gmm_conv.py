from typing import Tuple, Union

import torch
from torch import Tensor
from torch.nn import Parameter

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size

from ..inits import glorot, zeros


class GMMConv(MessagePassing):
    r"""The gaussian mixture model convolutional operator from the `"Geometric
    Deep Learning on Graphs and Manifolds using Mixture Model CNNs"
    <https://arxiv.org/abs/1611.08402>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \frac{1}{|\mathcal{N}(i)|}
        \sum_{j \in \mathcal{N}(i)} \frac{1}{K} \sum_{k=1}^K
        \mathbf{w}_k(\mathbf{e}_{i,j}) \odot \mathbf{\Theta}_k \mathbf{x}_j,

    where

    .. math::
        \mathbf{w}_k(\mathbf{e}) = \exp \left( -\frac{1}{2} {\left(
        \mathbf{e} - \mathbf{\mu}_k \right)}^{\top} \Sigma_k^{-1}
        \left( \mathbf{e} - \mathbf{\mu}_k \right) \right)

    denotes a weighting function based on trainable mean vector
    :math:`\mathbf{\mu}_k` and diagonal covariance matrix
    :math:`\mathbf{\Sigma}_k`.

    .. note::

        The edge attribute :math:`\mathbf{e}_{ij}` is usually given by
        :math:`\mathbf{e}_{ij} = \mathbf{p}_j - \mathbf{p}_i`, where
        :math:`\mathbf{p}_i` denotes the position of node :math:`i` (see
        :class:`torch_geometric.transform.Cartesian`).

    Args:
        in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.
        out_channels (int): Size of each output sample.
        dim (int): Pseudo-coordinate dimensionality.
        kernel_size (int): Number of kernels :math:`K`.
        separate_gaussians (bool, optional): If set to :obj:`True`, will
            learn separate GMMs for every pair of input and output channel,
            inspired by traditional CNNs. (default: :obj:`False`)
        aggr (str, optional): The aggregation operator to use
            (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
            (default: :obj:`"mean"`)
        root_weight (bool, optional): If set to :obj:`False`, the layer will
            not add transformed root node features to the output.
            (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})` or
          :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
          if bipartite,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge features :math:`(|\mathcal{E}|, D)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})` or
          :math:`(|\mathcal{V}_t|, F_{out})` if bipartite
    """
    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, dim: int, kernel_size: int,
                 separate_gaussians: bool = False, aggr: str = 'mean',
                 root_weight: bool = True, bias: bool = True, **kwargs):
        super().__init__(aggr=aggr, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dim = dim
        self.kernel_size = kernel_size
        self.separate_gaussians = separate_gaussians
        self.root_weight = root_weight

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)
        self.rel_in_channels = in_channels[0]

        if in_channels[0] > 0:
            self.g = Parameter(
                Tensor(in_channels[0], out_channels * kernel_size))

            if not self.separate_gaussians:
                self.mu = Parameter(Tensor(kernel_size, dim))
                self.sigma = Parameter(Tensor(kernel_size, dim))
            if self.separate_gaussians:
                self.mu = Parameter(
                    Tensor(in_channels[0], out_channels, kernel_size, dim))
                self.sigma = Parameter(
                    Tensor(in_channels[0], out_channels, kernel_size, dim))
        else:
            self.g = torch.nn.parameter.UninitializedParameter()
            self.mu = torch.nn.parameter.UninitializedParameter()
            self.sigma = torch.nn.parameter.UninitializedParameter()
            self._hook = self.register_forward_pre_hook(
                self.initialize_parameters)

        if root_weight:
            self.root = Linear(in_channels[1], out_channels, bias=False,
                               weight_initializer='glorot')

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        if not isinstance(self.g, torch.nn.UninitializedParameter):
            glorot(self.g)
            glorot(self.mu)
            glorot(self.sigma)
        if self.root_weight:
            self.root.reset_parameters()
        zeros(self.bias)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None):
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor, edge_attr: OptTensor)
        if not self.separate_gaussians:
            out: OptPairTensor = (torch.matmul(x[0], self.g), x[1])
            out = self.propagate(edge_index, x=out, edge_attr=edge_attr,
                                 size=size)
        else:
            out = self.propagate(edge_index, x=x, edge_attr=edge_attr,
                                 size=size)

        x_r = x[1]
        if x_r is not None and self.root is not None:
            out = out + self.root(x_r)

        if self.bias is not None:
            out = out + self.bias

        return out

    def message(self, x_j: Tensor, edge_attr: Tensor):
        EPS = 1e-15
        F, M = self.rel_in_channels, self.out_channels
        (E, D), K = edge_attr.size(), self.kernel_size

        if not self.separate_gaussians:
            gaussian = -0.5 * (edge_attr.view(E, 1, D) -
                               self.mu.view(1, K, D)).pow(2)
            gaussian = gaussian / (EPS + self.sigma.view(1, K, D).pow(2))
            gaussian = torch.exp(gaussian.sum(dim=-1))  # [E, K]

            return (x_j.view(E, K, M) * gaussian.view(E, K, 1)).sum(dim=-2)

        else:
            gaussian = -0.5 * (edge_attr.view(E, 1, 1, 1, D) -
                               self.mu.view(1, F, M, K, D)).pow(2)
            gaussian = gaussian / (EPS + self.sigma.view(1, F, M, K, D).pow(2))
            gaussian = torch.exp(gaussian.sum(dim=-1))  # [E, F, M, K]

            gaussian = gaussian * self.g.view(1, F, M, K)
            gaussian = gaussian.sum(dim=-1)  # [E, F, M]

            return (x_j.view(E, F, 1) * gaussian).sum(dim=-2)  # [E, M]

    @torch.no_grad()
    def initialize_parameters(self, module, input):
        if isinstance(self.g, torch.nn.parameter.UninitializedParameter):
            x = input[0][0] if isinstance(input, tuple) else input[0]
            in_channels = x.size(-1)
            out_channels, kernel_size = self.out_channels, self.kernel_size
            self.g.materialize((in_channels, out_channels * kernel_size))
            if not self.separate_gaussians:
                self.mu.materialize((kernel_size, self.dim))
                self.sigma.materialize((kernel_size, self.dim))
            else:
                self.mu.materialize(
                    (in_channels, out_channels, kernel_size, self.dim))
                self.sigma.materialize(
                    (in_channels, out_channels, kernel_size, self.dim))
            glorot(self.g)
            glorot(self.mu)
            glorot(self.sigma)

        module._hook.remove()
        delattr(module, '_hook')

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, dim={self.dim})')
