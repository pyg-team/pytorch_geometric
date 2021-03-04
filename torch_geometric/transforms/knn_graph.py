import torch_geometric
from torch_geometric.utils import to_undirected


class KNNGraph(object):
    r"""Creates a k-NN graph based on node positions :obj:`pos`.

    Args:
        k (int, optional): The number of neighbors. (default: :obj:`6`)
        loop (bool, optional): If :obj:`True`, the graph will contain
            self-loops. (default: :obj:`False`)
        force_undirected (bool, optional): If set to :obj:`True`, new edges
            will be undirected. (default: :obj:`False`)
        flow (string, optional): The flow direction when using in combination
            with message passing (:obj:`"source_to_target"` or
            :obj:`"target_to_source"`). (default: :obj:`"source_to_target"`)
    """
    def __init__(self, k=6, loop=False, force_undirected=False,
                 flow='source_to_target'):

        self.k = k
        self.loop = loop
        self.force_undirected = force_undirected
        self.flow = flow

    def __call__(self, data):
        data.edge_attr = None
        batch = data.batch if 'batch' in data else None
        edge_index = torch_geometric.nn.knn_graph(data.pos, self.k, batch,
                                                  loop=self.loop,
                                                  flow=self.flow)

        if self.force_undirected:
            edge_index = to_undirected(edge_index, num_nodes=data.num_nodes)

        data.edge_index = edge_index

        return data

    def __repr__(self):
        return '{}(k={})'.format(self.__class__.__name__, self.k)
