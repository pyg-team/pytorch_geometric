import torch

from torch_geometric.data import Data
from torch_geometric.transforms import LineGraph


def test_line_graph():
    transform = LineGraph()
    assert str(transform) == 'LineGraph()'

    # Directed.
    edge_index = torch.tensor([
        [0, 1, 2, 2, 3],
        [1, 2, 0, 3, 0],
    ])
    data = Data(edge_index=edge_index, num_nodes=4)
    data = transform(data)
    assert data.edge_index.tolist() == [[0, 1, 1, 2, 3, 4], [1, 2, 3, 0, 4, 0]]
    assert data.num_nodes == data.edge_index.max().item() + 1

    # Directed with unsorted `edge_index` and `edge_attr`:
    edge_index = torch.tensor([[1, 0, 2], [2, 1, 0]])
    edge_attr = torch.tensor([[12.], [1.], [20.]])
    data = Data(edge_index=edge_index, edge_attr=edge_attr, num_nodes=3)
    data = transform(data)
    assert data.edge_index.tolist() == [[0, 1, 2], [1, 2, 0]]
    assert data.x.tolist() == [[1.], [12.], [20.]]

    # Directed with duplicated edges:
    edge_index = torch.tensor([[0, 0, 1], [1, 1, 2]])
    edge_attr = torch.tensor([[1.], [2.], [3.]])
    data = Data(edge_index=edge_index, edge_attr=edge_attr, num_nodes=3)
    data = transform(data)
    assert data.num_nodes == 2
    assert data.x.size(0) == data.num_nodes

    # Undirected with `force_directed=True` and unsorted `edge_index`:
    edge_index = torch.tensor([[1, 0], [0, 1]])
    edge_attr = torch.tensor([[10.], [1.]])
    data = Data(edge_index=edge_index, edge_attr=edge_attr, num_nodes=2)
    data = LineGraph(force_directed=True)(data)
    assert data.x.tolist() == [[1.], [10.]]

    # Undirected.
    edge_index = torch.tensor([[0, 0, 0, 1, 1, 2, 2, 3, 3, 3, 4, 4],
                               [1, 2, 3, 0, 4, 0, 3, 0, 2, 4, 1, 3]])
    edge_attr = torch.ones(edge_index.size(1))
    data = Data(edge_index=edge_index, edge_attr=edge_attr, num_nodes=5)
    data = transform(data)
    assert data.edge_index.max().item() + 1 == data.x.size(0)
    assert data.edge_index.tolist() == [
        [0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 4, 4, 4, 5, 5, 5],
        [1, 2, 3, 0, 2, 4, 0, 1, 4, 5, 0, 5, 1, 2, 5, 2, 3, 4],
    ]
    assert data.x.tolist() == [2, 2, 2, 2, 2, 2]
    assert data.num_nodes == data.edge_index.max().item() + 1
