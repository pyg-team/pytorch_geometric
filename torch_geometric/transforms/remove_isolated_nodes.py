import re

import torch
from torch_geometric.utils import remove_isolated_nodes


class RemoveIsolatedNodes(object):
    r"""Removes isolated nodes from the graph."""

    def __call__(self, data):
        num_nodes = data.num_nodes
        out = remove_isolated_nodes(data.edge_index, data.edge_attr, num_nodes)
        data.edge_index, data.edge_attr, mask = out

        for key, item in data:
            if bool(re.search('edge', key)):
                continue
            if torch.is_tensor(item) and item.size(0) == num_nodes:
                data[key] = item[mask]

        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)
