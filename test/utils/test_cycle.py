import torch

from torch_geometric.utils import find_all_cycles, find_max_weight_cycle


def test_find_max_weight_cycle_simple():
    # Simple triangle cycle: 0 -> 1 -> 2 -> 0
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    edge_weight = torch.tensor([1.0, 2.0, 3.0])

    cycle, weight = find_max_weight_cycle(edge_index, edge_weight)

    assert cycle is not None
    assert cycle.tolist() == [0, 1, 2]
    assert weight == 6.0


def test_find_max_weight_cycle_no_cycle():
    # DAG with no cycles
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
    edge_weight = torch.tensor([1.0, 2.0, 3.0])

    cycle, weight = find_max_weight_cycle(edge_index, edge_weight)

    assert cycle is None
    assert weight == float('-inf')


def test_find_max_weight_cycle_multiple_cycles():
    # Graph with two cycles:
    # Cycle 1: 0 -> 1 -> 0 (weight: 1 + 2 = 3)
    # Cycle 2: 2 -> 3 -> 2 (weight: 4 + 5 = 9)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]])
    edge_weight = torch.tensor([1.0, 2.0, 4.0, 5.0])

    cycle, weight = find_max_weight_cycle(edge_index, edge_weight)

    assert cycle is not None
    assert weight == 9.0
    # Should find the heavier cycle
    assert 2 in cycle.tolist() and 3 in cycle.tolist()


def test_find_max_weight_cycle_negative_weights():
    # DAG with reversed edges (negative weights)
    # Forward: 0 -> 1 (5), 1 -> 2 (3), 2 -> 3 (2)
    # Backward: 1 -> 0 (-4), 2 -> 1 (-1)
    # Cycle 0 -> 1 -> 0: 5 + (-4) = 1
    # Cycle 0 -> 1 -> 2 -> 1 -> 0: 5 + 3 + (-1) + (-4) = 3
    edge_index = torch.tensor([[0, 1, 2, 1, 2], [1, 2, 3, 0, 1]])
    edge_weight = torch.tensor([5.0, 3.0, 2.0, -4.0, -1.0])

    cycle, weight = find_max_weight_cycle(edge_index, edge_weight)

    assert cycle is not None
    assert weight > 0  # Should find a positive weight cycle


def test_find_max_weight_cycle_self_loop():
    # Graph with self-loop
    edge_index = torch.tensor([[0, 0, 1], [0, 1, 0]])
    edge_weight = torch.tensor([10.0, 1.0, 2.0])

    cycle, weight = find_max_weight_cycle(edge_index, edge_weight)

    # Should find the 0 -> 1 -> 0 cycle, not the self-loop
    assert cycle is not None
    assert len(cycle) > 1  # Not a self-loop


def test_find_max_weight_cycle_no_weights():
    # Test with default weights (all 1.0)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])

    cycle, weight = find_max_weight_cycle(edge_index)

    assert cycle is not None
    assert cycle.tolist() == [0, 1, 2]
    assert weight == 3.0


def test_find_max_weight_cycle_max_length():
    # Test max_cycle_length parameter
    # Create a long cycle: 0 -> 1 -> 2 -> 3 -> 4 -> 0
    edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]])
    edge_weight = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])

    # With max_cycle_length=3, should not find the 5-node cycle
    cycle, weight = find_max_weight_cycle(
        edge_index,
        edge_weight,
        max_cycle_length=3,
    )

    # Should either find no cycle or a shorter one
    if cycle is not None:
        assert len(cycle) <= 3


def test_find_max_weight_cycle_top_k():
    # Test top_k_paths parameter with complex graph
    edge_index = torch.tensor([
        [0, 0, 1, 1, 2, 2, 3],
        [1, 2, 2, 3, 3, 0, 0],
    ])
    edge_weight = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])

    # Should work with different top_k values
    cycle1, weight1 = find_max_weight_cycle(
        edge_index,
        edge_weight,
        top_k_paths=10,
    )
    cycle2, weight2 = find_max_weight_cycle(
        edge_index,
        edge_weight,
        top_k_paths=100,
    )

    # Both should find a cycle
    assert cycle1 is not None
    assert cycle2 is not None


def test_find_all_cycles_simple():
    # Graph with two simple cycles
    edge_index = torch.tensor([[0, 1, 2, 1], [1, 2, 0, 0]])
    edge_weight = torch.tensor([1.0, 1.0, 1.0, 2.0])

    cycles, weights = find_all_cycles(edge_index, edge_weight)

    assert len(cycles) >= 2  # At least two cycles
    assert len(cycles) == len(weights)

    # Check that we found the expected cycles
    cycle_sets = [set(c.tolist()) for c in cycles]
    assert {0, 1, 2} in cycle_sets  # Triangle cycle
    assert {0, 1} in cycle_sets  # Two-node cycle


def test_find_all_cycles_min_weight():
    # Test min_weight filtering
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]])
    edge_weight = torch.tensor([1.0, 2.0, 4.0, 5.0])

    # Only cycles with weight >= 5 should be returned
    cycles, weights = find_all_cycles(edge_index, edge_weight, min_weight=5.0)

    assert all(w >= 5.0 for w in weights)


def test_find_all_cycles_no_cycles():
    # DAG with no cycles
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])

    cycles, weights = find_all_cycles(edge_index)

    assert len(cycles) == 0
    assert len(weights) == 0


def test_find_all_cycles_max_length():
    # Test max_cycle_length parameter
    edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]])

    cycles, weights = find_all_cycles(edge_index, max_cycle_length=3)

    # All cycles should have length <= 3
    assert all(len(c) <= 3 for c in cycles)


def test_cycle_detection_large_graph():
    # Test with a larger graph
    num_nodes = 20
    # Create a ring graph with some additional edges
    edge_list = [[i, (i + 1) % num_nodes] for i in range(num_nodes)]
    # Add some shortcuts
    edge_list.extend([[0, 5], [5, 10], [10, 15], [15, 0]])

    edge_index = torch.tensor(edge_list).t()
    edge_weight = torch.ones(edge_index.size(1))

    cycle, weight = find_max_weight_cycle(
        edge_index,
        edge_weight,
        max_cycle_length=10,
    )

    # Should find some cycle
    assert cycle is not None
    assert weight > 0


def test_cycle_detection_disconnected_graph():
    # Test with disconnected components
    # Component 1: 0 -> 1 -> 0
    # Component 2: 2 -> 3 -> 2
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]])
    edge_weight = torch.tensor([1.0, 1.0, 5.0, 5.0])

    cycle, weight = find_max_weight_cycle(edge_index, edge_weight)

    # Should find the heavier cycle in component 2
    assert cycle is not None
    assert weight == 10.0


def test_cycle_detection_cuda():
    if not torch.cuda.is_available():
        return

    # Test on CUDA device
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]]).cuda()
    edge_weight = torch.tensor([1.0, 2.0, 3.0]).cuda()

    cycle, weight = find_max_weight_cycle(edge_index, edge_weight)

    assert cycle is not None
    assert cycle.device.type == 'cuda'
    assert weight == 6.0
