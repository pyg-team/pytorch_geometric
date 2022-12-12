from typing import Callable, Optional

from torch_geometric.utils import barabasi_albert_graph

from .graph_generator import GraphGenerator


class BAGraph(GraphGenerator):
    r"""Generate base graph for Barabasi-Albert (BA) graph.

    Args:
        num_nodes (int, optional): Specifies the number of the nodes in the
            base graph. (default: :obj:`300`)
        num_edges: number of edges
        num_motifs: number of motifs to attach to the base graph
        motif (Motif, optional): the Motif object
    """
    def __init__(
        self,
        num_nodes: int = 300,
        num_edges: int = 5,
        num_motifs: int = 80,
        motif: Optional[Callable] = None,
        seed: int = None,
    ):
        self.num_edges = num_edges
        self.num_motifs = num_motifs
        super().__init__(num_nodes, motif, seed)

    def generate_base_graph(self) -> None:
        self._edge_index = barabasi_albert_graph(self.num_nodes,
                                                 self.num_edges)
        self.attach_motif(self.num_motifs)
        self.generate_feature(num_features=10)
