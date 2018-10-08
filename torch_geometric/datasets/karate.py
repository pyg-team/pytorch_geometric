import torch
import networkx as nx
from torch_geometric.data import InMemoryDataset, Data


class KarateClub(InMemoryDataset):
    url = 'http://faust.is.tue.mpg.de/'

    def __init__(self, transform=None):
        super(KarateClub, self).__init__('.', transform, None, None)

        adj = nx.to_scipy_sparse_matrix(nx.karate_club_graph()).tocoo()
        row = torch.from_numpy(adj.row).to(torch.long)
        col = torch.from_numpy(adj.col).to(torch.long)
        edge_index = torch.stack([row, col], dim=0)
        self.data, self.slices = self.collate([Data(edge_index=edge_index)])

    def _download(self):
        return

    def _process(self):
        return

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)
