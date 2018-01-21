import torch
from torch.nn import Module, Parameter

from .utils.inits import uniform


class MoConv(Module):
    """MoNet Convolutional Operator.

    Args:
        in_features (int): Size of each input sample.
        out_features (int): Size of each output sample.
        dim (int): Pseudo-coordinate dimensionality.
        kernel_size (int): Size of the convolving kernel.
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
    """

    def __init__(self, in_features, out_features, dim, kernel_size, bias=True):

        super(MoConv, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.dim = dim
        self.kernel_size = kernel_size
        self.mean = Parameter(
            torch.Tensor(out_features, in_features, kernel_size, dim))
        self.covariance = Parameter(
            torch.Tensor(out_features, in_features, kernel_size, dim))
        self.root = Parameter(torch.Tensor(out_features, in_features))

        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        size = self.in_features * self.kernel_size * self.dim
        uniform(size, self.mean)
        uniform(size, self.covariance)
        uniform(size, self.root)
        uniform(size, self.bias)

    def forward(self, adj, input):
        pass

    def __repr__(self):
        return repr(self, ['kernel_size'])
