import networkx as nx
import numpy as np
import torch

from torch_geometric.utils import calculate_ppr


def test_graph_ibmb():
    G = nx.karate_club_graph()
    edge_index = torch.tensor(list(G.edges)).t()
    num_nodes = len(G.nodes)

    neighbors, weights = calculate_ppr(edge_index, 0.1, 1.e-5,
                                       num_nodes=num_nodes,
                                       target_indices=np.array([0, 4, 5, 6]))
    assert len(neighbors) == 4

    neighbors, weights = calculate_ppr(
        edge_index, 0.5, 1.e-5, num_nodes=num_nodes,
        target_indices=np.array([0, 4, 5, 6, 10]))
    assert len(neighbors) == 5

    neighbors, weights = calculate_ppr(edge_index, 0.5, 1.e-5,
                                       num_nodes=num_nodes,
                                       target_indices=None)
    assert len(neighbors) == 34
