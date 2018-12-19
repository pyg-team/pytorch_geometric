import torch
from torch_geometric.utils import scatter_


class MessagePassing(torch.nn.Module):
    def __init__(self, aggr='add'):
        super(MessagePassing, self).__init__()

        assert aggr in ['add', 'mean', 'max']
        self.aggr = aggr

    def message(self, x_i, x_j, **kwargs):
        return x_j

    def update(self, x_old, x):
        return x

    def forward(self, x, edge_index, edge_attr=None, **kwargs):
        kwargs['x'] = x
        (row, col) = edge_index

        gathered_dict = {}
        for key, value in kwargs.items():
            value = value.unsqueeze(-1) if value.dim() == 1 else value
            if value.size(0) == x.size(0):
                gathered_dict['{}_i'.format(key)] = value[row]
                gathered_dict['{}_j'.format(key)] = value[col]
            else:
                gathered_dict[key] = value

        gathered_dict['edge_index'] = edge_index
        if edge_attr is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.unsqueeze(-1)
            gathered_dict['edge_attr'] = edge_attr

        out = self.message(**gathered_dict)
        h = scatter_(self.aggr, out, row, dim_size=x.size(0))
        out = self.update(x, h)

        return out

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)
