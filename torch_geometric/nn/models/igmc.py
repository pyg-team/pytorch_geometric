from typing import Union

import torch

from torch_geometric.data import Data, HeteroData


def igmc_node_label(batch: Union[Data, HeteroData], num_sampled_edges: int):

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

        return torch.tensor(label_list)
