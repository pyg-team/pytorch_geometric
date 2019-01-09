import torch
from torch_sparse import spspmm, coalesce

from torch_geometric.utils import remove_self_loops


class TwoHop(object):
    r"""Adds the two hop edges to the edge indices."""

    def __call__(self, data):
        edge_index, edge_attr = data.edge_index, data.edge_attr
        n = data.num_nodes

        fill = 1e16
        value = edge_index.new_full(
            (edge_index.size(1), ), fill, dtype=torch.float)

        index, value = spspmm(edge_index, value, edge_index, value, n, n, n)
        index, value = remove_self_loops(index, value)

        edge_index = torch.cat([edge_index, index], dim=1)
        if edge_attr is None:
            data.edge_index, _ = coalesce(edge_index, None, n, n)
        else:
            value = value.view(-1, *[1 for _ in range(edge_attr.dim() - 1)])
            value = value.expand(-1, *list(edge_attr.size())[1:])
            edge_attr = torch.cat([edge_attr, value], dim=0)
            data.edge_index, edge_attr = coalesce(
                edge_index, edge_attr, n, n, op='min', fill_value=fill)
            edge_attr[edge_attr >= fill] = 0
            data.edge_attr = edge_attr

        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)
