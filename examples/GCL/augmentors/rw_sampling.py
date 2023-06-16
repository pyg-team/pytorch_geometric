import torch
from GCL.augmentors.augmentor import Augmentor, Graph
from torch_sparse import SparseTensor

from torch_geometric.utils import subgraph, to_edge_index
from torch_geometric.utils.num_nodes import maybe_num_nodes


class RWSampling(Augmentor):
    def __init__(self, num_seeds: int, walk_length: int):
        super(RWSampling, self).__init__()
        self.num_seeds = num_seeds
        self.walk_length = walk_length

    def augment(self, g: Graph) -> Graph:
        x, edge_index, edge_weights = g.unfold()
        num_nodes = maybe_num_nodes(edge_index)
        start = torch.randint(0, num_nodes, size=(self.num_seeds, ),
                              dtype=torch.long).to(x.device)
        node_idx = edge_index.random_walk(start.flatten(),
                                          self.walk_length).view(-1)
        #node_idx = random_walk(edge_index, start.flatten(), self.walk_length).view(-1)

        edge_index, _ = to_edge_index(edge_index)
        edge_index, edge_weight = subgraph(node_idx, edge_index)
        sparse_edge_index = SparseTensor(row=edge_index[0], col=edge_index[1],
                                         sparse_sizes=[num_nodes, num_nodes])

        return Graph(x=x, adj_t=sparse_edge_index, edge_weights=edge_weights)
