from typing import Callable

from torch_geometric.data import Data
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
    """

    def __init__(
        self,
        motif: Callable,
        num_nodes: int,
        edge_prob: float = 5,
        num_motifs: int = 80,
    ):
        self.edge_prob = edge_prob
        self.num_motifs = num_motifs
        super().__init__(motif, num_nodes)

    def generate_base_graph(self):
        self.edge_index = erdos_renyi_graph(self.num_nodes, self.edge_prob)
        self.attach_motif()
        self.generate_feature()

        data = Data(
            x=self.x,
            edge_index=self.edge_index,
            y=self.node_label,
            expl_mask=self.expl_mask,
            edge_label=self.edge_label,
        )

        return data
