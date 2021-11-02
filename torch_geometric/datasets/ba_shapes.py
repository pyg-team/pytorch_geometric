from typing import Optional, Callable

import torch
import numpy as np
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import barabasi_albert_graph


def house():
    edge_index = [[0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 4],
                  [1, 3, 4, 4, 2, 0, 1, 3, 2, 0, 0, 1]]
    edge_index = torch.tensor(edge_index)
    label = torch.tensor([1, 1, 2, 2, 3])
    return edge_index, label


class BAShapes(InMemoryDataset):
    r"""The BA-Shapes dataset from the `"GNNExplainer: Generating Explanations
    for Graph Neural Networks" <https://arxiv.org/pdf/1903.03894.pdf>`_ paper,
    containing a Barabasi-Albert (BA) graph with 300 nodes and a set of 80
    "house"-structured graphs connected to it.

    Args:
        connection_distribution (string, optional): Specifies how the houses
            and the BA graph get connected. Valid inputs are :obj:`random`
            (random BA graph nodes are selected for connection to the houses),
            :obj:`uniform` (uniformly distributed BA graph nodes are selected
            for connection to the houses). (default: :obj:`random`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
    """
    def __init__(self, connection_distribution: str = 'random',
                 transform: Optional[Callable] = None):
        super().__init__('.', transform)
        assert connection_distribution in ['random', 'uniform']

        # Build Barabasi Albert Braph
        num_nodes = 300
        edge_index = barabasi_albert_graph(num_nodes=num_nodes, num_edges=5)
        label = torch.zeros(num_nodes, dtype=int)
        edge_label = torch.zeros(edge_index.size(1), dtype=int)

        # Select nodes to connect shapes
        num_shapes = 80
        if connection_distribution == 'random':
            connecting_nodes = np.random.choice(num_nodes, num_shapes, False)
        else:
            step = num_nodes // num_shapes
            connecting_nodes = np.arange(0, num_nodes, step=step)

        # Connect shapes to basis graph
        for i in range(num_shapes):
            house_edge_index, house_label = house()
            house_edge_index = house_edge_index + num_nodes
            connecting_edge_index = torch.tensor([
                [connecting_nodes[i], num_nodes],
                [num_nodes, connecting_nodes[i]]]
            )
            edge_index = torch.cat([
                edge_index, house_edge_index, connecting_edge_index], dim=-1)
            edge_label = torch.cat([
                edge_label,
                torch.tensor(house_edge_index.size(1) * [1]),
                torch.tensor([0, 0])]
            )
            label = torch.cat((label, house_label))
            num_nodes += 5

        x = torch.ones((num_nodes, 10), dtype=torch.float)
        expl_mask = np.zeros(num_nodes, dtype=bool)
        expl_mask[range(400, num_nodes, 5)] = True

        data = Data(x=x, edge_index=edge_index, y=label, expl_mask=expl_mask,
                    edge_label=edge_label)

        self.data, self.slices = self.collate([data])
