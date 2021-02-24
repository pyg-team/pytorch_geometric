from typing import Union, Tuple, Optional, Callable
from torch_geometric.typing import PairTensor, Adj, OptTensor

import torch
from torch import Tensor
from torch.nn import Parameter
from torch.nn import Linear
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax

from ..inits import zeros


class GatedGraphConv2(MessagePassing):
    r"""The gated graph convolutional operator from the
    `"Residual Gated Graph ConNets"
    <https://arxiv.org/abs/1711.07553>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = 
        \mathbf{U} \mathbf{x}_i +
        \sum_{j \in \mathcal{N}(i)} \boldsymbol{\eta}_{ij}
        \odot \mathbf{V} \mathbf{x}_j

    where the gates :math:`\boldsymbol{\eta}_{ij}` are defined by 
    
    .. math::
        \boldsymbol{\eta}_{ij} = \sigma(\mathbf{A} \mathbf{x}_i + 
        \mathbf{B} \mathbf{x}_j)
    
    :math:`\sigma` designates the sigmoid function.
    
    Args:
        in_channels (int or tuple): Size of each input sample. A tuple
            corresponds to the sizes of source and target dimensionalities.
        out_channels (int): Size of each output sample.
        act (callable, optional): Gating function :math:`\sigma`.
            (default: :meth:`torch.sigmoid`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        root_weight (bool, optional): If set to :obj:`False`, the layer will
            not add transformed root node features to the output.
            (default: :obj:`True`)
        aggr (string, optional): The aggregation scheme to use
            (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
            (default: :obj:`"add"`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            act: Optional[Callable] = torch.sigmoid,
            root_weight: bool = True,
            bias: bool = True,
            aggr: str = 'add',
            **kwargs):
        super(GatedGraphConv2, self).__init__(aggr=aggr, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act = act
        
        if root_weight:
            self.lin_skip = Linear(in_channels,out_channels, bias=False)
        else:
            self.register_parameter('lin_skip', None)
        
        if bias:
            self.bias = Parameter(Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.lin_key   = Linear(in_channels, out_channels)
        self.lin_query = Linear(in_channels, out_channels)
        self.lin_value = Linear(in_channels, out_channels)
        
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()
        if self.lin_skip is not None:
            self.lin_skip.reset_parameters()
        if self.bias is not None:
            zeros(self.bias)
        
    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        """"""
        
        k = self.lin_key(x)
        q = self.lin_query(x)
        v = self.lin_value(x)
        
        out = self.propagate(edge_index, k=k, q=q, v=v)
        
        if self.lin_skip is not None:
           out += self.lin_skip(x)
        
        if self.bias is not None:
            out += self.bias
        
        return out

    def message(self, k_i: Tensor, q_j: Tensor, v_j: Tensor) -> Tensor:
        out = self.act(k_i + q_j) * v_j
        return out

    def __repr__(self):
        return '{}({}, {}, root_weight={}, bias={})'.format(self.__class__.__name__,
                                                     self.in_channels,
                                                     self.out_channels,
                                                     self.root_weight is not None,
                                                     self.bias is not None)
