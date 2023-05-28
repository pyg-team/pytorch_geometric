from torch_geometric.datasets.graph_generator import ERGraph


def test_er_graph():
    graph_generator = ERGraph(num_nodes=300, edge_prob=0.1)
    assert str(graph_generator) == 'ERGraph(num_nodes=300, edge_prob=0.1)'

    data = graph_generator()
    assert len(data) == 2
    assert data.num_nodes == 300
    assert data.num_edges >= 300 * 300 * 0.05
    assert data.num_edges <= 300 * 300 * 0.15
