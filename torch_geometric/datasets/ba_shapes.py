from typing import Optional, Callable

import torch
import numpy as np
import networkx as nx
from torch_geometric.data import InMemoryDataset, Data


def house():
    G = nx.house_graph()
    labels = [1, 1, 2, 2, 3]
    attr = {i: labels[i] for i in range(5)}
    nx.set_node_attributes(G, attr, 'label')
    nx.set_edge_attributes(G, 1, 'label')
    return G


def ba(num_nodes=300, m=5, seed=None):
    G = nx.barabasi_albert_graph(num_nodes, m, seed=seed)
    nx.set_node_attributes(G, 0, 'label')
    nx.set_edge_attributes(G, 0, 'label')
    return G


class BAShapes(InMemoryDataset):
    r"""The BA-Shapes dataset from the `"GNNExplainer: Generating Explanations
    for Graph Neural Networks" <https://arxiv.org/pdf/1903.03894.pdf>`_ paper,
    containing a Barabasi-Albert (BA) graph with 300 nodes and a set of 80
    "house"-structured graphs connected to it.

    Args:
        seed (int, optional): Random seed used for :obj:`networkx` Barabási
            Albert graph generation and for :obj:`numpy` methods.
            (default: :obj:`None`)
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
    def __init__(
        self,
        seed: Optional[int] = None,
        connection_distribution: str = 'random',
        transform: Optional[Callable] = None,
    ):
        super().__init__('.', transform, None, None)
        assert connection_distribution in ['random', 'uniform']
        if seed is not None:
            np.random.seed(seed)

        # Build Barabasi Albert Braph
        num_nodes = 300
        connecting_degree = 5
        G = ba(num_nodes, connecting_degree, seed)

        # Select nodes to connect shapes
        num_shapes = 80
        if connection_distribution == 'random':
            connecting_nodes = np.random.choice(num_nodes, num_shapes, False)
        else:
            step = num_nodes // num_shapes
            connecting_nodes = np.arange(0, num_nodes, step=step)

        # Connect shapes to basis graph
        for i in range(num_shapes):
            G_house = house()
            G_house = nx.relabel_nodes(
                G_house, {
                    2: num_nodes,
                    3: num_nodes + 1,
                    0: num_nodes + 3,
                    1: num_nodes + 2,
                    4: num_nodes + 4
                })
            G = nx.union(G, G_house)
            G.add_edge(num_nodes, connecting_nodes[i], label=0)
            num_nodes += 5

        # Edge label mask
        G = G.to_directed() if not nx.is_directed(G) else G
        edge_labels = list(nx.get_edge_attributes(G, 'label').values())
        edge_labels = torch.LongTensor(edge_labels)

        num_nodes = len(G.nodes())
        x = torch.ones((num_nodes, 10), dtype=torch.float)
        edge_index = torch.LongTensor(list(G.edges)).t().contiguous()
        y = list(nx.get_node_attributes(G, 'label').values())
        y = torch.LongTensor(y)
        test_mask = np.zeros(num_nodes, dtype=bool)
        test_mask[range(400, num_nodes, 5)] = True

        data = Data(x=x, edge_index=edge_index, y=y, test_mask=test_mask,
                    edge_label=edge_labels)

        self.data, self.slices = self.collate([data])

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)
