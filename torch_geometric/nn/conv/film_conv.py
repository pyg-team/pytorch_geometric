from typing import Callable, Union, Tuple, Optional
from torch_geometric.typing import PairTensor, Adj, OptTensor, Size

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn.conv import MessagePassing

from ..inits import reset


class FiLMConv(MessagePassing):
    r"""The FiLM graph convolutional operator from the
    `"GNN-FiLM: Graph Neural Networks with Feature-wise Linear Modulation"
    <https://arxiv.org/abs/1906.12192>`_
    paper

    .. math::
        \mathbf{x}^{\prime}_i = \sum_{j \in \mathcal{N}(i)} \sigma \left(
        \boldsymbol{\gamma}_i \odot \mathbf{x}_j \mathbf{W} + \boldsymbol{\beta}_i
        \right)

    where :math:`\boldsymbol{\beta}_i, \boldsymbol{\gamma}_i = g(\mathbf{x}_i)`
    with :math:`g` a neural network. :math:`\sigma` is a non-linearity. 

    Args:
        in_channels (int or tuple): Size of each input sample. A tuple
            corresponds to the sizes of source and target dimensionalities.
        out_channels (int): Size of each output sample.
        nn (torch.nn.Module): A neural network :math:`g` that
            maps node features :obj:`x` of shape
            :obj:`[-1, in_channels]` to shape :obj:`[-1, 2 * out_channels]`,
            *e.g.*, defined by :class:`torch.nn.Sequential`.
        act (callable, optional): Activation function :math:`\sigma`.
            If set to :obj:`None`, the activation function is simply the
            identity funtion. A good choice is :meth:`torch.tanh`.
            (default: :obj:`None`)
        aggr (string, optional): The aggregation operator to use
            (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
            (default: :obj:`"mean"`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            the additive bias associated to :math:`\mathbf{W}`
            (not to be confounded with :math:`\boldsymbol{\beta}_i`).
            (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, nn: Callable, act:  Optional[Callable] = None,
                 aggr: str = 'mean', bias: bool = True, **kwargs):
        super(FiLMConv, self).__init__(aggr=aggr, **kwargs)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nn = nn
        self.act = act
        
        if isinstance(in_channels, int):
            self.in_channels = (in_channels, out_channels)
        
        self.lin = Linear(self.in_channels[0], self.out_channels, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        reset(self.nn)
        
    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
                size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: PairTensor = (x, x)
        
        beta, gamma = torch.split(self.nn(x[1]), self.out_channels, dim=1)
        out = self.propagate(edge_index, x=x, beta=beta, gamma=gamma)
        if self.act is None:
            out = gamma * out + beta
            
        return out

    def message(self, x_j, beta, gamma) -> Tensor:
        if self.act is None:
            return self.lin(x_j)
        else:
            return self.act( gamma * self.lin(x_j) + beta )

    def __repr__(self):
        return '{}({}, dim={})'.format(self.__class__.__name__, self.channels,
                                       self.dim)
