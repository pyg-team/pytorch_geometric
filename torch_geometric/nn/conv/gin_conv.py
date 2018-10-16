import torch.nn as nn
from torch_scatter import scatter_add


class GINConv(nn.Module):
    def __init__(self, nn):
        super(GINConv, self).__init__()
        self.nn = nn
        self.reset_parameters()

    def reset_parameters(self):
        for item in self.nn:
            if hasattr(item, 'reset_parameters'):
                item.reset_parameters()

    def forward(self, x, edge_index):
        row, col = edge_index
        out = scatter_add(x[col], row, dim=0, dim_size=x.size(0))
        out = out + x
        out = self.nn(out)
        return out

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.nn)
