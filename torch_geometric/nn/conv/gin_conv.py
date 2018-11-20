import torch.nn as nn
from torch_scatter import scatter_add

from ..inits import reset


class GINConv(nn.Module):
    def __init__(self, nn, add_root=True):
        super(GINConv, self).__init__()
        self.nn = nn
        self.add_root = add_root
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)

    def forward(self, x, edge_index, size=None):
        size = x.size(0) if size is None else size

        row, col = edge_index
        out = scatter_add(x[col], row, dim=0, dim_size=size)
        if self.add_root:
            out = out + x
        out = self.nn(out)
        return out

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.nn)
