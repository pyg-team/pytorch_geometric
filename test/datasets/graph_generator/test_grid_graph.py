from torch_geometric.datasets.graph_generator import GridGraph


def test_grid_graph():
    graph_generator = GridGraph(height=10, width=10)
    assert str(graph_generator) == 'GridGraph(height=10, width=10)'

    data = graph_generator()
    assert len(data) == 2
    assert data.num_nodes == 100
    assert data.num_edges == 784
