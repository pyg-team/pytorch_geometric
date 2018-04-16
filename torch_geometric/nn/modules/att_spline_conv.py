import torch
from torch.nn import Module, Parameter

from .utils.inits import uniform
from .utils.repeat import repeat_to
from ..functional.att_spline_conv import att_spline_conv


class AttSplineConv(Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 dim,
                 kernel_size,
                 is_open_spline=True,
                 degree=1,
                 negative_slope=0.2,
                 dropout=0.5,
                 bias=True):

        super(AttSplineConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.degree = degree
        self.negative_slope = negative_slope
        self.dropout = dropout

        kernel_size = torch.LongTensor(repeat_to(kernel_size, dim))
        self.register_buffer('kernel_size', kernel_size)

        is_open_spline = torch.ByteTensor(repeat_to(is_open_spline, dim))
        self.register_buffer('is_open_spline', is_open_spline)

        weight = torch.Tensor(kernel_size.prod(), in_channels, out_channels)
        self.weight = Parameter(weight)

        self.root_weight = Parameter(torch.Tensor(in_channels, out_channels))

        self.att_weight = Parameter(torch.Tensor(2 * out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        size = self.in_channels * self.weight.size(0)
        uniform(size, self.weight)
        uniform(size, self.root_weight)
        uniform(size, self.att_weight)
        uniform(size, self.bias)

    def forward(self, x, edge_index, pseudo):
        dropout = self.dropout if self.training else 0
        return att_spline_conv(x, edge_index, pseudo, self.weight,
                               self.root_weight, self.att_weight,
                               self._buffers['kernel_size'],
                               self._buffers['is_open_spline'], self.degree,
                               self.negative_slope, dropout, self.bias)
