import torch
import numpy as np
import networkx as nx
from torch_geometric.data import InMemoryDataset, Data


class KarateClub(InMemoryDataset):
    r"""Zachary's karate club network from the `"An Information Flow Model for
    Conflict and Fission in Small Groups"
    <http://www1.ind.ku.dk/complexLearning/zachary1977.pdf>`_ paper, containing
    34 nodes, connected by 154 (undirected and unweighted) edges.
    Every node is labeled by one of two classes.

    Args:
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
    """
    def __init__(self, transform=None):
        super(KarateClub, self).__init__('.', transform, None, None)

        G = nx.karate_club_graph()

        adj = nx.to_scipy_sparse_matrix(G).tocoo()
        row = torch.from_numpy(adj.row.astype(np.int64)).to(torch.long)
        col = torch.from_numpy(adj.col.astype(np.int64)).to(torch.long)
        edge_index = torch.stack([row, col], dim=0)
        data = Data(edge_index=edge_index)
        data.num_nodes = edge_index.max().item() + 1
        data.x = torch.eye(data.num_nodes, dtype=torch.float)
        y = [0 if G.nodes[i]['club'] == 'Mr. Hi' else 1 for i in G.nodes]
        data.y = torch.tensor(y)
        self.data, self.slices = self.collate([data])

    def _download(self):
        return

    def _process(self):
        return

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)
