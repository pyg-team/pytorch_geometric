from torch_geometric.utils import add_self_loops, coalesce


class AddSelfLoops(object):
    def __call__(self, data):
        edge_index = data.edge_index
        edge_index = add_self_loops(edge_index, num_nodes=data.num_nodes)
        edge_index, _ = coalesce(edge_index, num_nods=data.num_nodes)
        data.edge_index = edge_index
        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)
