"""Example: Max-Weight Simple Cycle Detection.

===========================================

This example demonstrates how to use the cycle detection utilities in PyTorch
Geometric to find maximum weight cycles in directed graphs.

Use Case: DAG with Reversed Edges
----------------------------------
A common scenario is starting with a Directed Acyclic Graph (DAG) with
non-negative edge weights, then adding reversed edges with negative weights.
This creates cycles, and we want to find the cycle with the highest total
weight.
"""

import torch

from torch_geometric.utils import find_all_cycles, find_max_weight_cycle

# Example 1: Simple Triangle Cycle
# =================================
print("Example 1: Simple Triangle Cycle")
print("-" * 50)

# Create a simple cycle: 0 -> 1 -> 2 -> 0
edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
edge_weight = torch.tensor([1.0, 2.0, 3.0])

cycle, weight = find_max_weight_cycle(edge_index, edge_weight)

print(f"Edge index:\n{edge_index}")
print(f"Edge weights: {edge_weight.tolist()}")
print(f"Found cycle: {cycle.tolist()}")
print(f"Cycle weight: {weight}")
print()

# Example 2: DAG with Reversed Edges (Original Use Case)
# =======================================================
print("Example 2: DAG with Reversed Edges")
print("-" * 50)

# Original DAG: 0 -> 1 -> 2 -> 3
# Add reversed edges: 1 -> 0, 2 -> 1
# This creates multiple cycles with different weights

# Forward edges (positive weights)
forward_edges = torch.tensor([[0, 1, 2], [1, 2, 3]])
forward_weights = torch.tensor([5.0, 3.0, 2.0])

# Reversed edges (negative weights)
reverse_edges = torch.tensor([[1, 2], [0, 1]])
reverse_weights = torch.tensor([-4.0, -1.0])

# Combine into single graph
edge_index = torch.cat([forward_edges, reverse_edges], dim=1)
edge_weight = torch.cat([forward_weights, reverse_weights])

print(f"Forward edges: {forward_edges.t().tolist()}")
print(f"Forward weights: {forward_weights.tolist()}")
print(f"Reverse edges: {reverse_edges.t().tolist()}")
print(f"Reverse weights: {reverse_weights.tolist()}")
print()

# Find the maximum weight cycle
cycle, weight = find_max_weight_cycle(edge_index, edge_weight)

if cycle is not None:
    print(f"Maximum weight cycle: {cycle.tolist()}")
    print(f"Cycle weight: {weight}")

    # Reconstruct the cycle path with edge weights
    print("\nCycle path:")
    for i in range(len(cycle)):
        src = cycle[i].item()
        dst = cycle[(i + 1) % len(cycle)].item()

        # Find edge weight
        edge_mask = (edge_index[0] == src) & (edge_index[1] == dst)
        edge_w = edge_weight[edge_mask].item()

        print(f"  {src} -> {dst}: weight = {edge_w}")
else:
    print("No cycle found")
print()

# Example 3: Finding All Cycles
# ==============================
print("Example 3: Finding All Cycles")
print("-" * 50)

# Create a graph with multiple cycles
edge_index = torch.tensor([
    [0, 1, 2, 1, 3, 4],  # Sources
    [1, 2, 0, 0, 4, 3],  # Targets
])
edge_weight = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

print(f"Edge index:\n{edge_index}")
print(f"Edge weights: {edge_weight.tolist()}")
print()

# Find all cycles
cycles, weights = find_all_cycles(edge_index, edge_weight)

print(f"Found {len(cycles)} cycles:")
for i, (cycle, weight) in enumerate(zip(cycles, weights)):
    print(f"  Cycle {i + 1}: {cycle.tolist()}, weight = {weight}")
print()

# Example 4: Filtering Cycles by Minimum Weight
# ==============================================
print("Example 4: Filtering Cycles by Minimum Weight")
print("-" * 50)

# Only find cycles with weight >= 5.0
cycles, weights = find_all_cycles(edge_index, edge_weight, min_weight=5.0)

print("Cycles with weight >= 5.0:")
for i, (cycle, weight) in enumerate(zip(cycles, weights)):
    print(f"  Cycle {i + 1}: {cycle.tolist()}, weight = {weight}")
print()

# Example 5: Limiting Cycle Length
# =================================
print("Example 5: Limiting Cycle Length")
print("-" * 50)

# Create a longer cycle
edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]])
edge_weight = torch.ones(5)

print("Graph has a 5-node cycle")
print()

# Search with different max lengths
for max_len in [3, 5, 10]:
    cycle, weight = find_max_weight_cycle(
        edge_index,
        edge_weight,
        max_cycle_length=max_len,
    )

    if cycle is not None:
        print(f"Max length {max_len}: Found cycle of length {len(cycle)}")
    else:
        print(f"Max length {max_len}: No cycle found")
print()

# Example 6: Performance Tuning with top_k_paths
# ===============================================
print("Example 6: Performance Tuning")
print("-" * 50)

# Create a dense graph
num_nodes = 10
edges = []
weights = []

# Create a grid-like structure with cycles
for i in range(num_nodes - 1):
    edges.append([i, i + 1])
    weights.append(1.0)
    if i > 0:
        edges.append([i, i - 1])
        weights.append(0.5)

# Add some long-range connections
edges.extend([[0, 5], [5, 9], [9, 0]])
weights.extend([2.0, 2.0, 2.0])

edge_index = torch.tensor(edges).t()
edge_weight = torch.tensor(weights)

print(f"Graph with {num_nodes} nodes and {len(edges)} edges")
print()

# Compare different top_k values
for top_k in [10, 50, 100]:
    cycle, weight = find_max_weight_cycle(
        edge_index,
        edge_weight,
        top_k_paths=top_k,
    )

    if cycle is not None:
        print(f"top_k={top_k}: Found cycle with weight {weight:.2f}")
    else:
        print(f"top_k={top_k}: No cycle found")

print()
print("=" * 50)
print("Examples completed!")
