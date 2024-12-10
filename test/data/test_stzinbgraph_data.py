import torch

from torch_geometric.data import STZINBGraph
from torch_geometric.loader import DataLoader


def test_stzinb_graph():
    # Test data
    od_pairs = [(0, 1), (1, 2), (2, 3)]  # Example O-D pairs
    demand = torch.rand(4, 10)  # 4 nodes (O-D pairs), 10 time windows
    time_windows = [
        "t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9", "t10"
    ]

    # Create an O-D graph
    od_graph = STZINBGraph.from_od_data(od_pairs, demand, time_windows)

    # Compute the number of unique nodes
    unique_nodes = set()
    for o, d in od_pairs:
        unique_nodes.add(o)
        unique_nodes.add(d)
    num_nodes = len(unique_nodes)

    # Assertions for edge_index
    assert od_graph.edge_index is not None, "Edge index should not be None."
    assert (od_graph.edge_index.size(0) == 2
            ), "Edge index should have two rows (source, target)."
    assert (od_graph.edge_index.size(1)
            > 0), "Edge index should have at least one edge."

    # Assertions for node features (x)
    assert od_graph.x is not None, "Node features (x) should not be None."
    assert (
        od_graph.x.size(0) == num_nodes
    ), f"Node features should match the number of unique nodes ({num_nodes})."
    assert (
        od_graph.x.size(1) == 1
    ), "Node features should have a single feature dim (aggregated demand)."

    # Assertions for temporal features (time_series)
    assert (od_graph.time_series
            is not None), "Temporal features (time_series) should not be None."
    assert (od_graph.time_series.size() == demand.size()
            ), "Temporal features should match the input demand matrix."

    # Assertions for adjacency matrix (adj)
    assert od_graph.adj is not None, (
        "Adjacency matrix (adj) should not be None.")

    assert (od_graph.adj.size(0) == num_nodes
            ), f"Adjacency matrix should be square ({num_nodes}x{num_nodes})."
    assert (od_graph.adj.size(1) == num_nodes
            ), f"Adjacency matrix should be square ({num_nodes}x{num_nodes})."
    assert torch.all(od_graph.adj.diagonal() ==
                     1), "Self-loops should exist (diagonal entries = 1)."

    # Integration with DataLoader
    graphs = [od_graph] * 5  # Create a small dataset with 5 identical graphs
    loader = DataLoader(graphs, batch_size=2, shuffle=True)

    for batch in loader:
        assert batch.batch is not None, (
            "Batch object should have a 'batch' attribute.")
        assert batch.x is not None, (
            "Batched data should have node features (x).")
        assert batch.edge_index is not None, (
            "Batched data should have edge indices.")
        assert batch.time_series is not None, (
            "Batched data should retain temporal features (time_series).")

    print("STZINBGraph test passed successfully!")


# Run the test
if __name__ == "__main__":
    test_stzinb_graph()
