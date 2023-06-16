from GCL.augmentors.augmentor import Graph, Augmentor
from torch_geometric.utils import dropout_edge,to_edge_index
from torch_geometric.utils.num_nodes import maybe_num_nodes

from torch_sparse import SparseTensor

class EdgeRemoving(Augmentor):
    def __init__(self, pe: float):
        super(EdgeRemoving, self).__init__()
        self.pe = pe

    def augment(self, g: Graph) -> Graph:
        x, edge_index, edge_weights = g.unfold()
        num_nodes = maybe_num_nodes(edge_index)
        edge_index, _ = to_edge_index(edge_index)
        edge_index, _ = dropout_edge(edge_index, p=self.pe)
        sparse_edge_index = SparseTensor(row = edge_index[0], col = edge_index[1],sparse_sizes=[num_nodes,num_nodes])
        return Graph(x=x, adj_t=sparse_edge_index, edge_weights=edge_weights)
