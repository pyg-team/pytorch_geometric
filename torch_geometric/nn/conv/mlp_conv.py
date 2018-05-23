import torch
from torch.nn import Module, Parameter

from ..inits import uniform
from ..functional.mlp_conv import mlp_conv
from torch.nn import Linear


class MLPConv(Module):
    """Convolution operator using a continuous kernel function computed by an
       MLP. Kernel function maps pseudo-coordinates (adj) to
       in_features x out_features weights for the feature aggregation.
       (adapted from Gilmer et al.: Neural Message Passing for Quantum
       Chemistry, https://arxiv.org/abs/1704.01212)

    Args:
        in_features (int): Size of each input sample.
        out_features (int): Size of each output sample.
        num_hidden_layers (int): Number of hidden layers in the kernel MLP
        num_neurons_in_hidden (int or [int]): Size of the hidden layers.
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
    """

    def __init__(self,
                 in_features,
                 out_features,
                 dim=2,
                 num_hidden_layers=2,
                 num_neurons_in_hidden=128,
                 bias=True):

        super(MLPConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_hidden_layers = num_hidden_layers
        self.num_neurons_in_hidden = num_neurons_in_hidden
        self.dim = dim

        num_features = [self.dim] + self.num_neurons_in_hidden + \
                       [self.out_features * self.in_features]

        self.linears = [
            Linear(num_features[i], num_features[i + 1]).cuda()
            for i, _ in enumerate(num_features[:-1])
        ]

        self.root_weight = Parameter(torch.Tensor(in_features, out_features))

        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        size = self.in_features
        uniform(size, self.weight)
        uniform(size, self.bias)

    def forward(self, adj, input):
        return mlp_conv(adj, input, self.linears, self.root_weight,
                        self.out_features, self.bias)
