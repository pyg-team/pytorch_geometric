import torch
from torch_geometric.transforms import LineGraph
from torch_geometric.data import Data


def test_line_graph():
    assert LineGraph().__repr__() == 'LineGraph'


def test_line_graph_directed():
    edge_index = torch.tensor([[0, 1, 2, 2, 3],
                               [1, 2, 0, 3, 0]], dtype=torch.long)

    expected_row = [0, 1, 1, 2, 3, 4]
    expected_col = [1, 2, 3, 0, 4, 0]

    data = Data(edge_index=edge_index)
    data = LineGraph()(data)
    print(data.edge_index[0].tolist())
    print(data.edge_index[1].tolist())
    assert data.edge_index[0].tolist() == expected_row
    assert data.edge_index[1].tolist() == expected_col


def test_line_graph_undirected():
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 0, 2, 3, 3, 0],
                               [1, 0, 2, 1, 0, 2, 3, 2, 0, 3]],
                              dtype=torch.long)

    expected_row = [0, 1, 0, 2, 0, 4, 1, 2, 1, 3, 2, 3, 2, 4, 3, 4]
    expected_col = [1, 0, 2, 0, 4, 0, 2, 1, 3, 1, 3, 2, 4, 2, 4, 3]

    data = Data(edge_index=edge_index)
    data = LineGraph()(data)
    assert data.edge_index[0].tolist() == expected_row
    assert data.edge_index[1].tolist() == expected_col
