from torch_geometric.data import Data
from torch_geometric.datasets.graph_generator import GraphGenerator
from torch_geometric.utils import erdos_renyi_graph


class ERGraph(GraphGenerator):
    r"""Generates random Erdos-Renyi (ER) graphs.
    See :meth:`~torch_geometric.utils.erdos_renyi_graph` for more information.

    Args:
        num_nodes (int): The number of nodes.
        edge_prob (float): Probability of an edge.
    """
    def __init__(self, num_nodes: int, edge_prob: float):
        super().__init__()
        self.num_nodes = num_nodes
        self.edge_prob = edge_prob

    def __call__(self) -> Data:
        edge_index = erdos_renyi_graph(self.num_nodes, self.edge_prob)
        return Data(num_nodes=self.num_nodes, edge_index=edge_index)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(num_nodes={self.num_nodes}, '
                f'edge_prob={self.edge_prob})')
