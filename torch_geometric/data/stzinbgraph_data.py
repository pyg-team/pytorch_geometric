import torch

from torch_geometric.data import Data


class STZINBGraph(Data):
    """A class to represent an Origin-Destination (O-D) graph with temporal
    travel demand information.

    Attributes:
        edge_index (torch.Tensor): Tensor defining the edges between O-D pairs.
        x (torch.Tensor): Node features representing O-D pairs.
        time_series (torch.Tensor): Temporal features related to travel demand
            across time windows.
        adj (torch.Tensor): Adjacency matrix for the O-D graph.
    """
    def __init__(self, edge_index=None, x=None, time_series=None, adj=None,
                 **kwargs):
        super().__init__(edge_index=edge_index, x=x, **kwargs)
        self.time_series = time_series
        self.adj = adj

    @staticmethod
    def from_od_data(od_pairs, demand, time_windows, fully_connected=True,
                     normalize_adj=True):
        """Builds an O-D graph from raw data including O-D pairs, demand
        matrices, and time windows.

        Args:
            od_pairs (list of tuples): List of O-D pairs (e.g., [(origin1,
                dest1), (origin2, dest2), ...]).
            demand (torch.Tensor): Demand matrix of shape [num_nodes,
                num_time_windows].
            time_windows (list): List of time window labels (e.g., ['t1',
                't2', ..., 'tn']).
            fully_connected (bool): If True, creates a fully connected graph.
                Defaults to True.
            normalize_adj (bool): If True, normalizes the adjacency matrix.
                Defaults to True.

        Returns:
            STZINBGraph: A graph object with node features, edges, adjacency,
                and temporal features.
        """
        # Determine the set of unique nodes
        unique_nodes = set()
        for o, d in od_pairs:
            unique_nodes.add(o)
            unique_nodes.add(d)
        unique_nodes = sorted(unique_nodes)
        num_nodes = len(unique_nodes)

        # Map original indices to continuous indices if necessary
        node_mapping = {node: idx for idx, node in enumerate(unique_nodes)}
        od_pairs_mapped = [(node_mapping[o], node_mapping[d])
                           for o, d in od_pairs]

        # Create edge_index
        if fully_connected:
            # Fully connected graph
            edge_index = torch.combinations(torch.arange(num_nodes), r=2).t()
            edge_index = torch.cat([edge_index, edge_index.flip(0)],
                                   dim=1)  # Symmetric edges
        else:
            # Custom graph based on O-D adjacency
            edges = [(o, d) for o, d in od_pairs_mapped]
            edge_index = torch.tensor(edges, dtype=torch.long).t()

        # Build adjacency matrix
        adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float)
        for o, d in od_pairs_mapped:
            adj[o, d] = 1  # Directed edge
            adj[d, o] = 1  # Undirected graph (symmetric adjacency matrix)

        # Add self-loops
        adj += torch.eye(num_nodes)

        # Normalize adjacency matrix (excluding self-loops)
        if normalize_adj:
            row_sums = adj.sum(
                dim=1, keepdim=True) - 1  # Subtract self-loop contribution
            row_sums[row_sums == 0] = 1  # Avoid division by zero
            non_diag_mask = ~torch.eye(num_nodes, dtype=torch.bool)
            adj[non_diag_mask] /= row_sums.expand_as(adj)[non_diag_mask]

        # Adjust demand to match the number of nodes
        if demand.size(0) != num_nodes:
            raise ValueError(
                f"The number of rows in the demand matrix ({demand.size(0)}) "
                f"must match the number of unique nodes ({num_nodes}).")

        # Node features: Aggregate demand over all time windows
        x = demand.mean(dim=1, keepdim=True)  # Example: mean demand per node

        # Temporal features: Full time-series demand
        time_series = demand

        return STZINBGraph(edge_index=edge_index, x=x, time_series=time_series,
                           adj=adj)
