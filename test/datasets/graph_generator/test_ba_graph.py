from torch_geometric.datasets.graph_generator import BAGraph


def test_ba_graph():
    graph_generator = BAGraph(num_nodes=300, num_edges=5)
    data = graph_generator()
    assert len(data) == 2
    assert data.num_nodes == 300
    assert data.num_edges <= 2 * 300 * 5
