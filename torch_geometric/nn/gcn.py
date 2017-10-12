import torch
from torch.nn import Module, Parameter


class GCN(Module):
    """
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to False, the layer will not learn an additive bias.
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GCN, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))

        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        pass

    def forward(self, adj, features):
        pass

    def __repr__(self):
        s = ('{name}({in_channels}, {out_channels}')
        if self.bias is None:
            s += 'bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)
