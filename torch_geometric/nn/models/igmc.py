from typing import Optional, Union

import torch

from torch_geometric.data import Data, HeteroData
from torch_geometric.typing import NodeType


def igmc_node_label(batch: Union[Data, HeteroData], num_sampled_edges: int,
                    edge_direction: Optional[NodeType] = None):

    if isinstance(batch, Data):

        label_list = [float('inf')] * len(batch.batch)
        for i in range(2):
            for j in range(num_sampled_edges):
                label_list[(i * num_sampled_edges) + j] = i

        for i in range(len(batch.edge_index[0])):
            v = batch.edge_index[0][i]
            u = batch.edge_index[1][i]
            if v > u:
                label_list[v] = min(label_list[u] + 2, label_list[v])

        batch.igmc_id = torch.tensor(label_list)

    if isinstance(batch, HeteroData):

        assert edge_direction is not None

        origin = edge_direction[0]
        for node_type in batch.node_types:

            if node_type == origin:
                start_id = 0
            else:
                start_id = 1

            igmc_id = 0 + start_id
            label_list = [float('inf')] * len(batch[node_type].batch)
            prev = 0

            for index, i in enumerate(batch[node_type].batch):

                if i < prev:
                    igmc_id += 2
                label_list[index] = igmc_id
                prev = i

            batch[node_type].igmc_id = torch.tensor(label_list)

    return batch
