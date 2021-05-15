from typing import Optional
import math

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn.parameter import Parameter
try:
    from torch.nn.parameter import UninitializedParameter
except ImportError:
    # skip error for PyTorch 1.7 or before.
    pass

from torch_geometric.nn import inits


class Linear(torch.nn.Module):
    r"""
    A customized linear transformation similar to torch.nn.Linear.

    It support lazy initialization and customized initialization function.

    Args:
        in_features: size of each input sample, or -1 for lazy initialization.
        out_features: size of each output sample.
        bias: whether the layer learn the bias or not, default to False.
        weight_initializer: the initializer for the weight, currently support
         `glorot` and `kaiming_uniform`, default or not set will use
         `kaiming_uniform`.
        bias_initializer: the initializer for the bias, current support 'zeros'
         , or `constant (1 / math.sqrt(in_features)`, default or not set will
         use `constant`.
    """
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 weight_initializer: Optional[str] = None,
                 bias_initializer: Optional[str] = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer

        if in_features >= 0:
            self.weight = Parameter(torch.Tensor(out_features, in_features))
        else:
            self.weight = UninitializedParameter()
            self._hook = self.register_forward_pre_hook(
                self.initialize_parameters)
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.in_features < 0:
            return

        if self.weight_initializer == 'glorot':
            inits.glorot(self.weight)
        elif self.weight_initializer == 'kaiming_uniform' \
                or self.weight_initializer is None:
            # By default, we use kaiming uniform initializer.
            inits.kaiming_uniform(
                self.weight, fan=self.in_features, a=math.sqrt(5))
        else:
            raise RuntimeError(f"Linear layer weight initializer "
                               f"{self.weight_initializer} is not supported")

        if self.bias is not None:
            if self.bias_initializer == 'zeros':
                inits.zeros(self.bias)
            elif self.bias_initializer == 'constant' \
                    or self.bias_initializer is None:
                inits.normal(self.bias, 0, 1 / math.sqrt(self.in_features))
            else:
                raise RuntimeError(f"Linear layer bias initializer "
                                   f"{self.bias_initializer} is not supported")

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight, self.bias)

    @torch.no_grad()
    def initialize_parameters(self, module, input):
        if isinstance(self.weight, UninitializedParameter):
            self.in_features = input[0].size(-1)
            self.weight.materialize((self.out_features, self.in_features))
            self.reset_parameters()

        module._hook.remove()
        delattr(module, '_hook')

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_features}, '
                f'{self.out_features}, bias={self.bias is not None})')
