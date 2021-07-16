from torch_geometric.utils import to_undirected


class ToUndirected(object):
    r"""Converts the graph to an undirected graph, so that
    :math:`(j,i) \in \mathcal{E}` for every edge :math:`(i,j) \in \mathcal{E}`.

    Args:
        reduce (string, optional): The reduce operation to use for merging edge
            features. (default: :obj:`"add"`)
    """
    def __init__(self, reduce: str = "add"):
        self.reduce = reduce

    def __call__(self, data):
        if 'edge_index' in data:
            if 'edge_attr' in data:
                data.edge_index, data.edge_attr = to_undirected(
                    data.edge_index, data.edge_attr, num_nodes=data.num_nodes,
                    reduce=self.reduce)
            else:
                data.edge_index = to_undirected(data.edge_index,
                                                num_nodes=data.num_nodes)
        if 'adj_t' in data:
            data.adj_t = data.adj_t.to_symmetric(self.reduce)
        return data

    def __repr__(self):
        return f'{self.__class__.__name__}()'
