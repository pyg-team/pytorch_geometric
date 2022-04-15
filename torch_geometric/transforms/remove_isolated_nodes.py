import re
from typing import Union

import torch

from torch_geometric.data import Data, HeteroData
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import remove_isolated_nodes


@functional_transform('remove_isolated_nodes')
class RemoveIsolatedNodes(BaseTransform):
    r"""Removes isolated nodes from the graph
    (functional name: :obj:`remove_isolated_nodes`)."""
    def __call__(self, data: Union[Data,
                                   HeteroData]) -> Union[Data, HeteroData]:

        if isinstance(data, Data):
            num_nodes = data.num_nodes
            out = remove_isolated_nodes(data.edge_index, data.edge_attr,
                                        num_nodes)
            data.edge_index, data.edge_attr, mask = out

            if hasattr(data, '__num_nodes__'):
                data.num_nodes = int(mask.sum())

            for key, item in data:
                if bool(re.search('edge', key)):
                    continue
                if torch.is_tensor(item) and item.size(0) == num_nodes:
                    data[key] = item[mask]

            return data

        elif isinstance(data, HeteroData):
            return remove_hetero_isolated_nodes(data)

        raise TypeError(f"`RemoveIsolatedNodes` invalid type: {type(data)}")


def remove_hetero_isolated_nodes(data: HeteroData):
    r"""Removes the isolated nodes from the heterogenous graph
    given by :attr:`data`.
    Self-loops are preserved onlt for non-isolated nodes. Nodes with only
    a self loop are removed.

    Args:
        data (HeteroData): The graph to remove nodes from.

    :rtype: HeteroData
    """
    device = data[data.edge_types[0]].edge_index.device

    for node_type in data.node_types:

        num_nodes = data[node_type].num_nodes
        mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)

        for edge_type in data.edge_types:
            if 'edge_index' in data[edge_type]:
                edge_index = data[edge_type].edge_index
                if edge_type[0] == edge_type[-1] == node_type:
                    loop_mask = torch.where(edge_index[0] != edge_index[1])
                    mask[edge_index[0][loop_mask[0]]] = 1
                elif edge_type[0] == node_type:
                    mask[edge_index[0]] = 1
                elif edge_type[-1] == node_type:
                    mask[edge_index[1]] = 1

        assoc = torch.full((num_nodes, ), -1, dtype=torch.long,
                           device=mask.device)

        assoc[mask] = torch.arange(mask.sum(), device=assoc.device)

        for key, values in data[node_type].items():
            if bool(re.search('edge', key)):
                continue
            if torch.is_tensor(values) and values.size(
                    0) == data[node_type].num_nodes:
                data[node_type][key] = values[mask]

        for edge_type in data.edge_types:
            if 'edge_index' in data[edge_type]:
                edge_index = data[edge_type].edge_index

                if edge_type[0] == node_type:
                    edge_index[0] = assoc[edge_index[0]]
                if edge_type[-1] == node_type:
                    edge_index[1] = assoc[edge_index[1]]

                data[edge_type].edge_index = edge_index

            if edge_type[0] == edge_type[-1] == node_type:
                loop_mask = edge_index[0] != -1
                data[edge_type].edge_index = edge_index[:, loop_mask]
                if 'edge_attr' in data[edge_type]:
                    edge_attr = data[edge_type].edge_attr
                    print(edge_attr.size())
                    print(loop_mask.size())
                    data[edge_type].edge_attr = edge_attr[loop_mask]

    return data
