import networkx as nx

from torch_geometric.utils import from_networkx, to_undirected


def erdos_renyi_graph(n, p, seed=None, directed=False):
    r"""Returns the :obj:`edge_index` of a random Erdős-Rényi graph.

    Args:
        n (int): The number of nodes.
        p (float): Probability of an edge.
        seed (int, optional): Seed for random number generator.
            (default: :obj:`None`)
        directed (bool, optional): If set to :obj:`True`, will return a
            directed graph. (default: :obj:`False`)
    """

    G = nx.erdos_renyi_graph(n, p, seed, directed)
    edge_index = from_networkx(G).edge_index
    if not directed:
        edge_index = to_undirected(edge_index, n)
    return edge_index
