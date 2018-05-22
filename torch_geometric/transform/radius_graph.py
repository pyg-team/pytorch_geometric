from itertools import repeat

import torch
import scipy.spatial
from torch_geometric.utils import remove_self_loops


class RadiusGraph(object):
    def __init__(self, r):
        self.r = r

    def __call__(self, data):
        pos = data.pos
        assert not pos.is_cuda

        tree = scipy.spatial.cKDTree(pos)
        indices = tree.query_ball_tree(tree, self.r)

        row, col = [], []
        for i, neighbors in enumerate(indices):
            row += repeat(i, len(neighbors))
            col += neighbors
        edge_index = torch.tensor([row, col])
        edge_index, _ = remove_self_loops(edge_index)

        data.edge_index = edge_index
        return data

    def __repr__(self):
        return '{}(r={})'.format(self.__class__.__name__, self.r)
