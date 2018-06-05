import torch
import scipy.spatial
from torch_geometric.utils import to_undirected


class NNGraph(object):
    def __init__(self, k=6):
        self.k = k

    def __call__(self, data):
        pos = data.pos
        assert not pos.is_cuda

        row = torch.arange(pos.size(0), dtype=torch.long)
        row = row.view(-1, 1).repeat(1, self.k).view(-1)

        _, col = scipy.spatial.cKDTree(pos).query(pos, self.k + 1)
        col = torch.tensor(col)[:, 1:].contiguous().view(-1)
        mask = col < pos.size(0)
        edge_index = torch.stack([row[mask], col[mask]], dim=0)
        edge_index = to_undirected(edge_index, num_nodes=pos.size(0))

        data.edge_index = edge_index
        return data

    def __repr__(self):
        return '{}(k={})'.format(self.__class__.__name__, self.k)
