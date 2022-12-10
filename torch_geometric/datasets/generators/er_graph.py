from typing import Callable, Optional

from torch_geometric.utils import erdos_renyi_graph

from .graph_generator import GraphGenerator


class ERGraph(GraphGenerator):
    r"""Generate base graph for Erdos-Renyi (ER) graph.

    Args:
        motif: the Motif object
        num_nodes (int, optional): Specifies the number of the nodes in the
            base graph. (default: :obj:`300`)
        edge_prob: float: Probability of an edge.
        num_motifs: number of motifs to attach to the base graph
        seed: Seed to set randomness
    """
    def __init__(self, motif: Optional[Callable] = None, num_nodes: int = 300,
                 edge_prob: float = 5, num_motifs: int = 80, seed: int = None):
        self.edge_prob = edge_prob
        self.num_motifs = num_motifs
        super().__init__(num_nodes, motif, seed)

    def generate_base_graph(self):
        self._edge_index = erdos_renyi_graph(self.num_nodes, self.edge_prob)
        self.attach_motif()
        self.generate_feature()
