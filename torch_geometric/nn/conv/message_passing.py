import torch
from torch_geometric.utils import scatter_


class MessagePassing(torch.nn.Module):
    def __init__(self, aggr='add'):
        super(MessagePassing, self).__init__()

        assert aggr in ['add', 'mean', 'max']
        self.aggr = aggr

    def prepare(x, edge_index, edge_attr=None):
        return x, edge_index, edge_attr

    def message(x_i, x_j, edge_attr=None):
        return x_j

    def combine(x_old, x):
        return x

    def forward(self, x, edge_index, edge_attr=None):
        x = x.unsqueeze(-1) if x.dim() == 1 else x

        if edge_attr is not None and edge_attr.dim() == 1:
            edge_attr = edge_attr.view(-1, 1)

        x, edge_index, edge_attr = self.prepare(x, edge_index, edge_attr)

        row, col = edge_index
        out = self.messages(x[row], x[col], edge_attr)
        out = scatter_(self.aggr, out, row, dim_size=x.size(0))
        out = self.combine(x, out)

        return out

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)
