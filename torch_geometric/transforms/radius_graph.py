from itertools import repeat

import torch
import scipy.spatial
from torch_geometric.utils import remove_self_loops


class RadiusGraph(object):
    r"""Creates edges based on node positions to all points within a given
    distance.

    Args:
        r (float): The distance.
    """

    def __init__(self, r):
        self.r = r

    def __call__(self, data):
        pos = data.pos

        if pos.is_cuda:
            raise ValueError('The RadiusGraph transform is only implemented '
                             'for CPU')

        if 'batch' in data:
            raise ValueError('The RadiusGraph transform should not be used '
                             'for mini-batches')

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
