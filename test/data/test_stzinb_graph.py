import torch

from torch_geometric.data.stzinb_graph import STZINBGraph


def test_stzinb_graph():
    """Test the STZINBGraph class for building the O-D graph structure and
    validating its attributes.
    """
    # Example O-D pairs
    od_pairs = [(0, 1), (1, 2), (2, 3), (3, 0)]

    # Historical travel demand (num_time_windows x num_od_pairs)
    demand = torch.tensor([
        [10, 20, 15, 5],  # Time window 1
        [12, 18, 14, 6],  # Time window 2
        [9, 19, 16, 7]  # Time window 3
    ])

    # Time windows (e.g., 15-minute intervals)
    time_windows = [0, 15, 30]

    # Build the STZINBGraph
    graph = STZINBGraph.from_od_data(od_pairs, demand, time_windows)

    # Verify edge_index
    expected_edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])
    assert torch.equal(
        graph.edge_index,
        expected_edge_index), (f"Expected edge_index:\n{expected_edge_index}\n"
                               f"But got:\n{graph.edge_index}")

    # Verify node features (x)
    expected_x = torch.tensor([[10.3333], [19.0000], [15.0000], [6.0000]])
    assert torch.allclose(graph.x, expected_x,
                          atol=1e-4), (f"Expected x:\n{expected_x}\n"
                                       f"But got:\n{graph.x}")

    # Verify time_series
    expected_time_series = torch.tensor([
        [10, 12, 9],  # O-D pair (0 -> 1)
        [20, 18, 19],  # O-D pair (1 -> 2)
        [15, 14, 16],  # O-D pair (2 -> 3)
        [5, 6, 7]  # O-D pair (3 -> 0)
    ])
    assert torch.equal(graph.time_series, expected_time_series), (
        f"Expected time_series:\n{expected_time_series}\n"
        f"But got:\n{graph.time_series}")

    # Verify adjacency matrix (adj)
    expected_adj = torch.tensor([[0., 1., 0., 0.], [0., 0., 1., 0.],
                                 [0., 0., 0., 1.], [1., 0., 0., 0.]])
    assert torch.equal(graph.adj,
                       expected_adj), (f"Expected adj:\n{expected_adj}\n"
                                       f"But got:\n{graph.adj}")

    print("All tests passed for STZINBGraph!")


if __name__ == "__main__":
    test_stzinb_graph()
