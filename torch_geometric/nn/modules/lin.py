import torch
from torch.nn import Module, Parameter

from .utils.inits import uniform
from .utils.repr import repr
from ..functional.lin import lin


class Lin(Module):
    """Applies a 1x1 convolution over an input signal with input size
    :math:`(N, M_{in})` to output size :math:`(N, M_{out})`.

    Args:
        in_features (int): Size of each input sample.
        out_features (int): Size of each output sample.
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
    """

    def __init__(self, in_features, out_features, bias=True):
        super(Lin, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features, 1))

        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        size = self.in_features
        uniform(size, self.weight)
        uniform(size, self.bias)

    def forward(self, input):
        return lin(input, self.weight, self.bias)

    def __repr__(self):
        return repr(self)
