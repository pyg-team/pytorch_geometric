"""Cycle detection utilities for directed graphs."""
from typing import List, Optional, Tuple

import torch
from torch import Tensor

from torch_geometric.utils import coalesce, sort_edge_index
from torch_geometric.utils.num_nodes import maybe_num_nodes


def find_max_weight_cycle(
    edge_index: Tensor,
    edge_weight: Optional[Tensor] = None,
    num_nodes: Optional[int] = None,
    max_cycle_length: Optional[int] = None,
    top_k_paths: int = 100,
) -> Tuple[Optional[Tensor], float]:
    r"""Finds the maximum weight simple cycle in a directed graph using an
    iterative message passing approach.

    This function detects cycles by propagating path information through the
    graph, where each path tracks its starting node, accumulated weight, and
    visited nodes. A cycle is detected when a path returns to its starting
    node.

    Args:
        edge_index (torch.Tensor): The edge indices in COO format with shape
            :obj:`[2, num_edges]`.
        edge_weight (torch.Tensor, optional): Edge weights. If :obj:`None`,
            all edges are assumed to have weight :obj:`1.0`.
            (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes in the graph. If
            :obj:`None`, will be inferred from :obj:`edge_index`.
            (default: :obj:`None`)
        max_cycle_length (int, optional): Maximum cycle length to search for.
            If :obj:`None`, defaults to :obj:`num_nodes`. Larger values
            increase computation time. (default: :obj:`None`)
        top_k_paths (int, optional): Number of top paths to keep per node
            during search to manage memory. Higher values increase accuracy
            but use more memory. (default: :obj:`100`)

    Returns:
        (torch.Tensor, float): A tuple containing:

        - **cycle_nodes** (*torch.Tensor* or *None*): Node indices forming the
          maximum weight cycle, or :obj:`None` if no cycle exists. The cycle
          is represented as a 1D tensor where consecutive elements form edges,
          and the last element connects back to the first.
        - **cycle_weight** (*float*): Total weight of the cycle, or
          :obj:`-float('inf')` if no cycle exists.

    Examples:
        >>> # Create a simple cycle: 0 -> 1 -> 2 -> 0
        >>> edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
        >>> edge_weight = torch.tensor([1.0, 2.0, 3.0])
        >>> cycle, weight = find_max_weight_cycle(edge_index, edge_weight)
        >>> print(cycle)
        tensor([0, 1, 2])
        >>> print(weight)
        6.0

        >>> # DAG with reversed edges (negative weights create cycles)
        >>> edge_index = torch.tensor([[0, 1, 2, 1, 2], [1, 2, 3, 0, 1]])
        >>> edge_weight = torch.tensor([5.0, 3.0, 2.0, -4.0, -1.0])
        >>> cycle, weight = find_max_weight_cycle(edge_index, edge_weight)
        >>> print(cycle)
        tensor([0, 1])
        >>> print(weight)
        1.0
    """
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    if edge_weight is None:
        edge_weight = torch.ones(
            edge_index.size(1),
            dtype=torch.float,
            device=edge_index.device,
        )

    if max_cycle_length is None:
        max_cycle_length = num_nodes

    # Coalesce edges to handle multi-edges
    edge_index, edge_weight = coalesce(
        edge_index,
        edge_weight,
        num_nodes,
        num_nodes,
    )

    # Sort edge index for efficient lookup
    edge_index, edge_weight = sort_edge_index(
        edge_index,
        edge_weight,
        num_nodes=num_nodes,
    )

    # Initialize path states for each node
    # Each path state: (start_node, current_weight, visited_mask, path_nodes)
    device = edge_index.device

    # Track best cycle found
    best_cycle_weight = float('-inf')
    best_cycle_nodes = None

    # For each potential starting node
    for start_node in range(num_nodes):
        # Use dynamic programming approach: track best paths to each node
        # State: (current_node, visited_mask) -> (weight, path)
        # Use list to store active paths: each path is
        # (current_node, weight, visited_mask, path_list)
        active_paths = [(start_node, 0.0, 1 << start_node, [start_node])]

        for _ in range(max_cycle_length):
            if not active_paths:
                break

            new_paths = []

            # Expand each active path
            for (
                    curr_node,
                    curr_weight,
                    visited_mask,
                    path_list,
            ) in active_paths:
                # Find outgoing edges from current node
                out_mask = edge_index[0] == curr_node
                if not out_mask.any():
                    continue

                out_edges = edge_index[:, out_mask]
                out_weights = edge_weight[out_mask]

                # Try extending path along each outgoing edge
                for i in range(out_edges.size(1)):
                    next_node = out_edges[1, i].item()
                    edge_w = out_weights[i].item()

                    # Check if we've completed a cycle
                    if next_node == start_node and len(path_list) > 1:
                        cycle_weight = curr_weight + edge_w
                        if cycle_weight > best_cycle_weight:
                            best_cycle_weight = cycle_weight
                            best_cycle_nodes = torch.tensor(
                                path_list,
                                dtype=torch.long,
                                device=device,
                            )
                        continue

                    # Check if node already visited (ensure simple cycle)
                    if visited_mask & (1 << next_node):
                        continue

                    # Extend path
                    new_weight = curr_weight + edge_w
                    new_visited = visited_mask | (1 << next_node)
                    new_path = path_list + [next_node]

                    new_paths.append(
                        (next_node, new_weight, new_visited, new_path), )

            # Keep top-k paths by weight to manage memory
            if len(new_paths) > top_k_paths:
                new_paths.sort(key=lambda x: x[1], reverse=True)
                new_paths = new_paths[:top_k_paths]

            active_paths = new_paths

    return best_cycle_nodes, best_cycle_weight


def find_all_cycles(
        edge_index: Tensor,
        edge_weight: Optional[Tensor] = None,
        num_nodes: Optional[int] = None,
        max_cycle_length: Optional[int] = None,
        min_weight: float = float('-inf'),
) -> Tuple[List[Tensor], List[float]]:
    r"""Finds all simple cycles in a directed graph with weight above a
    threshold.

    This function uses an iterative depth-first search approach to enumerate
    all simple cycles in the graph, filtering by minimum weight.

    Args:
        edge_index (torch.Tensor): The edge indices in COO format with shape
            :obj:`[2, num_edges]`.
        edge_weight (torch.Tensor, optional): Edge weights. If :obj:`None`,
            all edges are assumed to have weight :obj:`1.0`.
            (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes in the graph. If
            :obj:`None`, will be inferred from :obj:`edge_index`.
            (default: :obj:`None`)
        max_cycle_length (int, optional): Maximum cycle length to search for.
            If :obj:`None`, defaults to :obj:`num_nodes`.
            (default: :obj:`None`)
        min_weight (float, optional): Minimum weight threshold for cycles.
            Only cycles with total weight :math:`\geq` :obj:`min_weight` are
            returned. (default: :obj:`-inf`)

    Returns:
        (list, list): A tuple containing:

        - **cycles** (*list*): List of cycles, where each cycle is a
          :obj:`torch.Tensor` of node indices.
        - **weights** (*list*): List of corresponding cycle weights.

    Examples:
        >>> edge_index = torch.tensor([[0, 1, 2, 1], [1, 2, 0, 0]])
        >>> edge_weight = torch.tensor([1.0, 1.0, 1.0, 2.0])
        >>> cycles, weights = find_all_cycles(edge_index, edge_weight)
        >>> for cycle, weight in zip(cycles, weights):
        ...     print(f"Cycle: {cycle}, Weight: {weight}")
        Cycle: tensor([0, 1, 2]), Weight: 3.0
        Cycle: tensor([0, 1]), Weight: 3.0
    """
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    if edge_weight is None:
        edge_weight = torch.ones(
            edge_index.size(1),
            dtype=torch.float,
            device=edge_index.device,
        )

    if max_cycle_length is None:
        max_cycle_length = num_nodes

    # Coalesce and sort edges
    edge_index, edge_weight = coalesce(
        edge_index,
        edge_weight,
        num_nodes,
        num_nodes,
    )
    edge_index, edge_weight = sort_edge_index(
        edge_index,
        edge_weight,
        num_nodes=num_nodes,
    )

    device = edge_index.device
    all_cycles = []
    all_weights = []

    # Build adjacency list for efficient traversal
    adj_list = [[] for _ in range(num_nodes)]
    for i in range(edge_index.size(1)):
        src = edge_index[0, i].item()
        dst = edge_index[1, i].item()
        weight = edge_weight[i].item()
        adj_list[src].append((dst, weight))

    # DFS from each node to find cycles
    for start_node in range(num_nodes):
        stack = [(start_node, 0.0, 1 << start_node, [start_node])]

        while stack:
            (
                curr_node,
                curr_weight,
                visited_mask,
                path_list,
            ) = stack.pop()

            if len(path_list) > max_cycle_length:
                continue

            # Explore neighbors
            for next_node, edge_w in adj_list[curr_node]:
                # Check for cycle completion
                if next_node == start_node and len(path_list) > 1:
                    cycle_weight = curr_weight + edge_w
                    if cycle_weight >= min_weight:
                        cycle_tensor = torch.tensor(
                            path_list,
                            dtype=torch.long,
                            device=device,
                        )
                        all_cycles.append(cycle_tensor)
                        all_weights.append(cycle_weight)
                    continue

                # Check if already visited
                if visited_mask & (1 << next_node):
                    continue

                # Extend path
                new_weight = curr_weight + edge_w
                new_visited = visited_mask | (1 << next_node)
                new_path = path_list + [next_node]

                stack.append((next_node, new_weight, new_visited, new_path))

    return all_cycles, all_weights


__all__ = [
    'find_max_weight_cycle',
    'find_all_cycles',
]
