from torch_geometric.data import Data
from torch_geometric.datasets.graph_generator import GraphGenerator
from torch_geometric.utils import barabasi_albert_graph


class BAGraph(GraphGenerator):
    r"""Generates random Barabasi-Albert (BA) graphs.
    See :meth:`~torch_geometric.utils.barabasis_albert_graph` for more
    information.

    Args:
        num_nodes (int): The number of nodes.
        num_edges (int): The number of edges from a new node to existing nodes.
    """
    def __init__(self, num_nodes: int, num_edges: int):
        self.num_nodes = num_nodes
        self.num_edges = num_edges

    def __call__(self) -> Data:
        edge_index = barabasi_albert_graph(self.num_nodes, self.num_edges)
        return Data(num_nodes=self.num_nodes, edge_index=edge_index)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(num_nodes={self.num_nodes}, '
                f'num_edges={self.num_edges})')
