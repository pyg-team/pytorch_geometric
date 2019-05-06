from torch_sparse import coalesce
from torch_geometric.utils import add_self_loops


class AddSelfLoops(object):
    r"""Adds self-loops to edge indices."""

    def __call__(self, data):
        N = data.num_nodes
        edge_index = data.edge_index
        assert data.edge_attr is None

        edge_index, _ = add_self_loops(edge_index, num_nodes=N)
        edge_index, _ = coalesce(edge_index, None, N, N)
        data.edge_index = edge_index
        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)
