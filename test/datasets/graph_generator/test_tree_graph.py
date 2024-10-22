import pytest

from torch_geometric.datasets.graph_generator import TreeGraph


@pytest.mark.parametrize('undirected', [False, True])
def test_tree_graph(undirected):
    graph_generator = TreeGraph(depth=2, branch=2, undirected=undirected)
    assert str(graph_generator) == (f'TreeGraph(depth=2, branch=2, '
                                    f'undirected={undirected})')

    data = graph_generator()
    assert len(data) == 3
    assert data.num_nodes == 7
    assert data.depth.tolist() == [0, 1, 1, 2, 2, 2, 2]
    if not undirected:
        assert data.edge_index.tolist() == [
            [0, 0, 1, 1, 2, 2],
            [1, 2, 3, 4, 5, 6],
        ]
    else:
        assert data.edge_index.tolist() == [
            [0, 0, 1, 1, 1, 2, 2, 2, 3, 4, 5, 6],
            [1, 2, 0, 3, 4, 0, 5, 6, 1, 1, 2, 2],
        ]
