from torch_geometric.utils import to_undirected


class ToUndirected(object):
    r"""Converts the graph to an undirected graph, so that
    :math:`(j,i) \in \mathcal{E}` for every edge :math:`(i,j) \in \mathcal{E}`.
    """
    def __call__(self, data):
        if data.edge_index is not None:
            data.edge_index = to_undirected(data.edge_index, data.num_nodes)
        if data.adj_t is not None:
            data.adj_t = data.adj_t.to_symmetric()
        return data

    def __repr__(self):
        return f'{self.__class__.__name__}()'
