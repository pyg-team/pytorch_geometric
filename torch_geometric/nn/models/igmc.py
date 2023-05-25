from typing import Union

import torch

from torch_geometric.typing import EdgeType, NodeType


def igmc_node_label(batch: Union[NodeType, dict[NodeType]],
                    edge_index: Union[EdgeType,
                                      dict[EdgeType]], num_sampled_edges: int):

    if isinstance(edge_index, torch.Tensor):

        label_list = [float('inf')] * len(batch)
        for i in range(2):
            for j in range(num_sampled_edges):
                label_list[(i * num_sampled_edges) + j] = i

        for i in range(len(edge_index[0])):
            v = edge_index[0][i]
            u = edge_index[1][i]
            if v > u:
                label_list[v] = min(label_list[u] + 2, label_list[v])

        return torch.tensor(label_list)
