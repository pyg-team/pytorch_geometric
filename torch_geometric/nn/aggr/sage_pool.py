import torch
from torch.nn import Parameter
import torch.nn.functional as F
import torch_scatter

from ..inits import uniform


class SAGEPoolAggr(torch.nn.Module):
    def __init__(self, in_channels, out_channels, op='max', bias=True):
        super(SAGEPoolAggr, self).__init__()
        assert op in ['max', 'mean', 'add']

        self.op = op
        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        size = self.weight.size(0)
        uniform(size, self.weight)
        uniform(size, self.bias)

    def forward(self, x, edge_index):
        row, col = edge_index

        out = torch.matmul(x, self.weight)

        if self.bias is not None:
            out = out + self.bias

        out = F.relu(out)

        op = getattr(torch_scatter, 'scatter_{}'.format(self.op))
        out = op(out[col], row, dim=0, dim_size=x.size(0))
        out = out[0] if isinstance(out, tuple) else out

        return out

    def __repr__(self):
        return '{}{}{}({}, {})'.format(self.__class__.__name__[:4],
                                       self.op.capitalize(),
                                       self.__class__.__name__[4:],
                                       self.weight.size(0),
                                       self.weight.size(1))
