import torch
from torch_scatter import scatter_add

from ..inits import reset


class GINConv(torch.nn.Module):
    def __init__(self, nn, eps=0, train=False):
        super(GINConv, self).__init__()
        self.nn = nn
        self.initial_eps = eps
        if train:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)
        self.eps.data.fill_(self.initial_eps)

    def forward(self, x, edge_index, size=None):
        size = x.size(0) if size is None else size

        row, col = edge_index
        out = scatter_add(x[col], row, dim=0, dim_size=size)
        out = (1 + self.eps) * x + out
        out = self.nn(out)
        return out

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.nn)
