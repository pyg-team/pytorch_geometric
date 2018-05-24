from torch_geometric.utils import add_self_loops


class AddSelfLoops(object):
    def __call__(self, data):
        data.edge_index = add_self_loops(data.edge_index, data.num_nodes)
        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)
