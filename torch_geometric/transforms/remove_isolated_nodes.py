import copy
from collections import defaultdict
from typing import Union

import torch

from torch_geometric.data import Data, HeteroData
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform


@functional_transform('remove_isolated_nodes')
class RemoveIsolatedNodes(BaseTransform):
    r"""Removes isolated nodes from the graph
    (functional name: :obj:`remove_isolated_nodes`).
    """
    def forward(
        self,
        data: Union[Data, HeteroData],
    ) -> Union[Data, HeteroData]:
        # Gather all nodes that occur in at least one edge (across all types):
        n_id_dict = defaultdict(list)
        for store in data.edge_stores:
            if 'edge_index' not in store:
                continue

            if store._key is None:
                src = dst = None
            else:
                src, _, dst = store._key

            n_id_dict[src].append(store.edge_index[0])
            n_id_dict[dst].append(store.edge_index[1])

        n_id_dict = {k: torch.cat(v).unique() for k, v in n_id_dict.items()}

        n_map_dict = {}
        for store in data.node_stores:
            if store._key not in n_id_dict:
                n_id_dict[store._key] = torch.empty((0, ), dtype=torch.long)

            idx = n_id_dict[store._key]
            mapping = idx.new_zeros(data.num_nodes)
            mapping[idx] = torch.arange(idx.numel(), device=mapping.device)
            n_map_dict[store._key] = mapping

        for store in data.edge_stores:
            if 'edge_index' not in store:
                continue

            if store._key is None:
                src = dst = None
            else:
                src, _, dst = store._key

            row = n_map_dict[src][store.edge_index[0]]
            col = n_map_dict[dst][store.edge_index[1]]
            store.edge_index = torch.stack([row, col], dim=0)

        old_data = copy.copy(data)
        for out, store in zip(data.node_stores, old_data.node_stores):
            for key, value in store.items():
                if key == 'num_nodes':
                    out.num_nodes = n_id_dict[store._key].numel()
                elif store.is_node_attr(key):
                    out[key] = value[n_id_dict[store._key]]

        return data
