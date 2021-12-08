from typing import Optional

import copy
import math

import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from torch_geometric.nn import inits


class Linear(torch.nn.Module):
    r"""Applies a linear tranformation to the incoming data

    .. math::
        \mathbf{x}^{\prime} = \mathbf{x} \mathbf{W}^{\top} + \mathbf{b}

    similar to :class:`torch.nn.Linear`.
    It supports lazy initialization and customizable weight and bias
    initialization.

    Args:
        in_channels (int): Size of each input sample. Will be initialized
            lazily in case it is given as :obj:`-1`.
        out_channels (int): Size of each output sample.
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        weight_initializer (str, optional): The initializer for the weight
            matrix (:obj:`"glorot"`, :obj:`"uniform"`, :obj:`"kaiming_uniform"`
            or :obj:`None`).
            If set to :obj:`None`, will match default weight initialization of
            :class:`torch.nn.Linear`. (default: :obj:`None`)
        bias_initializer (str, optional): The initializer for the bias vector
            (:obj:`"zeros"` or :obj:`None`).
            If set to :obj:`None`, will match default bias initialization of
            :class:`torch.nn.Linear`. (default: :obj:`None`)
    """
    def __init__(self, in_channels: int, out_channels: int, bias: bool = True,
                 weight_initializer: Optional[str] = None,
                 bias_initializer: Optional[str] = None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer

        if in_channels > 0:
            self.weight = Parameter(torch.Tensor(out_channels, in_channels))
        else:
            self.weight = nn.parameter.UninitializedParameter()
            self._hook = self.register_forward_pre_hook(
                self.initialize_parameters)

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self._load_hook = self._register_load_state_dict_pre_hook(
            self._lazy_load_hook)

        self.reset_parameters()

    def __deepcopy__(self, memo):
        out = Linear(self.in_channels, self.out_channels, self.bias
                     is not None, self.weight_initializer,
                     self.bias_initializer)
        if self.in_channels > 0:
            out.weight = copy.deepcopy(self.weight, memo)
        if self.bias is not None:
            out.bias = copy.deepcopy(self.bias, memo)
        return out

    def reset_parameters(self):
        if isinstance(self.weight, nn.parameter.UninitializedParameter):
            pass
        elif self.weight_initializer == 'glorot':
            inits.glorot(self.weight)
        elif self.weight_initializer == 'uniform':
            bound = 1.0 / math.sqrt(self.weight.size(-1))
            torch.nn.init.uniform_(self.weight.data, -bound, bound)
        elif self.weight_initializer == 'kaiming_uniform':
            inits.kaiming_uniform(self.weight, fan=self.in_channels,
                                  a=math.sqrt(5))
        elif self.weight_initializer is None:
            inits.kaiming_uniform(self.weight, fan=self.in_channels,
                                  a=math.sqrt(5))
        else:
            raise RuntimeError(f"Linear layer weight initializer "
                               f"'{self.weight_initializer}' is not supported")

        if isinstance(self.weight, nn.parameter.UninitializedParameter):
            pass
        elif self.bias is None:
            pass
        elif self.bias_initializer == 'zeros':
            inits.zeros(self.bias)
        elif self.bias_initializer is None:
            inits.uniform(self.in_channels, self.bias)
        else:
            raise RuntimeError(f"Linear layer bias initializer "
                               f"'{self.bias_initializer}' is not supported")

    def forward(self, x: Tensor) -> Tensor:
        """"""
        return F.linear(x, self.weight, self.bias)

    @torch.no_grad()
    def initialize_parameters(self, module, input):
        if isinstance(self.weight, nn.parameter.UninitializedParameter):
            self.in_channels = input[0].size(-1)
            self.weight.materialize((self.out_channels, self.in_channels))
            self.reset_parameters()
        self._hook.remove()
        delattr(self, '_hook')

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        if isinstance(self.weight, nn.parameter.UninitializedParameter):
            destination[prefix + 'weight'] = self.weight
        else:
            destination[prefix + 'weight'] = self.weight.detach()
        if self.bias is not None:
            destination[prefix + 'bias'] = self.bias.detach()

    def _lazy_load_hook(self, state_dict, prefix, local_metadata, strict,
                        missing_keys, unexpected_keys, error_msgs):

        weight = state_dict[prefix + 'weight']
        if isinstance(weight, nn.parameter.UninitializedParameter):
            self.in_channels = -1
            self.weight = nn.parameter.UninitializedParameter()
            if not hasattr(self, '_hook'):
                self._hook = self.register_forward_pre_hook(
                    self.initialize_parameters)

        elif isinstance(self.weight, nn.parameter.UninitializedParameter):
            self.in_channels = weight.size(-1)
            self.weight.materialize((self.out_channels, self.in_channels))
            if hasattr(self, '_hook'):
                self._hook.remove()
                delattr(self, '_hook')

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, bias={self.bias is not None})')


class HeteroLinear(torch.nn.Module):
    r"""Applies separate linear tranformations to the incoming data according
    to node types

    .. math::
        \mathbf{x}^{\prime}_{\kappa} = \mathbf{x}_{\kappa}
        \mathbf{W}^{\top}_{\kappa} + \mathbf{b}_{\kappa}

    for node type :math:`\kappa`.
    It supports lazy initialization and customizable weight and bias
    initialization.

    Args:
        in_channels (int): Size of each input sample. Will be initialized
            lazily in case it is given as :obj:`-1`.
        out_channels (int): Size of each output sample.
        num_node_types (int): The number of node types.
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.Linear`.
    """
    def __init__(self, in_channels: int, out_channels: int,
                 num_node_types: int, **kwargs):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.lins = torch.nn.ModuleList([
            Linear(in_channels, out_channels, **kwargs)
            for _ in range(num_node_types)
        ])

        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x: Tensor, node_type: Tensor) -> Tensor:
        """"""
        out = x.new_empty(x.size(0), self.out_channels)
        for i, lin in enumerate(self.lins):
            mask = node_type == i
            out[mask] = lin(x[mask])
        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, bias={self.lins[0].bias is not None})')
