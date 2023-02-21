from typing import Callable, Optional

import torch

from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.deprecation import deprecated
from torch_geometric.utils import barabasi_albert_graph


def house():
    edge_index = torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 4],
                               [1, 3, 4, 4, 2, 0, 1, 3, 2, 0, 0, 1]])
    label = torch.tensor([1, 1, 2, 2, 3])
    return edge_index, label


@deprecated("use 'datasets.ExplainerDataset' in combination with "
            "'datasets.graph_generator.BAGraph' instead")
class BAShapes(InMemoryDataset):
    r"""The BA-Shapes dataset from the `"GNNExplainer: Generating Explanations
    for Graph Neural Networks" <https://arxiv.org/abs/1903.03894>`__ paper,
    containing a Barabasi-Albert (BA) graph with 300 nodes and a set of 80
    "house"-structured graphs connected to it.

    .. warning::

        :class:`BAShapes` is deprecated and will be removed in a future
        release. Use :class:`ExplainerDataset` in combination with
        :class:`torch_geometric.datasets.graph_generator.BAGraph` instead.

    Args:
        connection_distribution (str, optional): Specifies how the houses
            and the BA graph get connected. Valid inputs are :obj:`"random"`
            (random BA graph nodes are selected for connection to the houses),
            and :obj:`"uniform"` (uniformly distributed BA graph nodes are
            selected for connection to the houses). (default: :obj:`"random"`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
    """
    def __init__(self, connection_distribution: str = "random",
                 transform: Optional[Callable] = None):
        super().__init__('.', transform)
        assert connection_distribution in ['random', 'uniform']

        # Build the Barabasi-Albert graph:
        num_nodes = 300
        edge_index = barabasi_albert_graph(num_nodes, num_edges=5)
        edge_label = torch.zeros(edge_index.size(1), dtype=torch.int64)
        node_label = torch.zeros(num_nodes, dtype=torch.int64)

        # Select nodes to connect shapes:
        num_houses = 80
        if connection_distribution == 'random':
            connecting_nodes = torch.randperm(num_nodes)[:num_houses]
        else:
            step = num_nodes // num_houses
            connecting_nodes = torch.arange(0, num_nodes, step)

        # Connect houses to Barabasi-Albert graph:
        edge_indices = [edge_index]
        edge_labels = [edge_label]
        node_labels = [node_label]
        for i in range(num_houses):
            house_edge_index, house_label = house()

            edge_indices.append(house_edge_index + num_nodes)
            edge_indices.append(
                torch.tensor([[int(connecting_nodes[i]), num_nodes],
                              [num_nodes, int(connecting_nodes[i])]]))

            edge_labels.append(
                torch.ones(house_edge_index.size(1), dtype=torch.long))
            edge_labels.append(torch.zeros(2, dtype=torch.long))

            node_labels.append(house_label)

            num_nodes += 5

        edge_index = torch.cat(edge_indices, dim=1)
        edge_label = torch.cat(edge_labels, dim=0)
        node_label = torch.cat(node_labels, dim=0)

        x = torch.ones((num_nodes, 10), dtype=torch.float)
        expl_mask = torch.zeros(num_nodes, dtype=torch.bool)
        expl_mask[torch.arange(400, num_nodes, 5)] = True

        data = Data(x=x, edge_index=edge_index, y=node_label,
                    expl_mask=expl_mask, edge_label=edge_label)

        self.data, self.slices = self.collate([data])
