import torch
import networkx as nx
from torch_sparse import SparseTensor
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
        super(KarateClub, self).__init__(transform=transform)

        G = nx.karate_club_graph()
        adj = nx.to_scipy_sparse_matrix(G)
        adj = SparseTensor.from_scipy(adj, has_value=False).fill_cache_()
        x = torch.eye(adj.size(0), dtype=torch.float)
        y = [0 if G.nodes[i]['club'] == 'Mr. Hi' else 1 for i in G.nodes]
        y = torch.tensor(y)

        data = Data(adj=adj, x=x, y=y)
        self.data, self.slices = self.collate([data])

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)
