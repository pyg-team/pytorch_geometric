import torch

from torch_geometric.data import Data


class STZINBGraph(Data):
    """Represents the Origin-Destination (O-D) graph with temporal
    travel demand information. Extends PyTorch Geometric's Data class.

    Attributes:
        edge_index (torch.Tensor): Defines the edges between
        O-D pairs as a tensor.
        x (torch.Tensor): Node features (O-D pairs) including
        historical travel demand values.
        time_series (torch.Tensor): Temporal features related to travel demand
            across multiple time windows.
        adj (torch.Tensor): Adjacency matrix for the O-D graph.

    Methods:
        from_od_data(od_pairs, demand, time_windows):
            Builds the graph structure from O-D pair data, demand matrices,
            and time windows.
    """
    def __init__(self, edge_index=None, x=None, time_series=None, adj=None):
        """Initializes the STZINBGraph class with optional attributes.

        Args:
            edge_index (torch.Tensor, optional): Edges between O-D pairs.
            x (torch.Tensor, optional): Node features for O-D pairs.
            time_series (torch.Tensor, optional): Temporal demand data for
                O-D pairs.
            adj (torch.Tensor, optional): Adjacency matrix for the O-D graph.
        """
        super().__init__()
        self.edge_index = edge_index
        self.x = x
        self.time_series = time_series
        self.adj = adj

    @classmethod
    def from_od_data(cls, od_pairs, demand, time_windows):
        """Constructs the O-D graph structure from O-D pair data,
        demand matrices, and time windows.

        Args:
            od_pairs (list of tuple): List of O-D pairs as
            (origin, destination).
            demand (torch.Tensor): Historical demand data of shape
                (num_time_windows, num_od_pairs).
            time_windows (list): List of time window identifiers.

        Returns:
            STZINBGraph: An instance of the STZINBGraph class.
        """
        # Build edge_index from O-D pairs
        origins, destinations = zip(*od_pairs)
        edge_index = torch.tensor(
            [list(origins), list(destinations)], dtype=torch.long)

        # Node features (average demand for each O-D pair)
        x = torch.mean(demand.float(),
                       dim=0).unsqueeze(1)  # Shape: (num_od_pairs, 1)

        # Temporal features
        time_series = demand.transpose(
            0, 1)  # Shape: (num_od_pairs, num_time_windows)

        # Adjacency matrix
        num_nodes = len(set(origins).union(set(destinations)))
        adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float)
        for origin, destination in od_pairs:
            adj[origin, destination] = 1

        return cls(edge_index=edge_index, x=x, time_series=time_series,
                   adj=adj)
