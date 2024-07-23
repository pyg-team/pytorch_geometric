import torch

from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import AddRemainingSelfLoops


def test_add_remaining_self_loops():
    assert str(AddRemainingSelfLoops()) == 'AddRemainingSelfLoops()'

    assert len(AddRemainingSelfLoops()(Data())) == 0

    # No self-loops in `edge_index`.
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    edge_weight = torch.tensor([1, 2, 3, 4])
    edge_attr = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]])

    data = Data(edge_index=edge_index, num_nodes=3)
    data = AddRemainingSelfLoops()(data)
    assert len(data) == 2
    assert data.edge_index.tolist() == [[0, 1, 1, 2, 0, 1, 2],
                                        [1, 0, 2, 1, 0, 1, 2]]
    assert data.num_nodes == 3

    # Single self-loop in `edge_index`.
    edge_index = torch.tensor([[0, 0, 1, 2], [1, 0, 2, 1]])
    data = Data(edge_index=edge_index, num_nodes=3)
    data = AddRemainingSelfLoops()(data)
    assert len(data) == 2
    assert data.edge_index.tolist() == [[0, 1, 2, 0, 1, 2], [1, 2, 1, 0, 1, 2]]
    assert data.num_nodes == 3

    data = Data(edge_index=edge_index, edge_weight=edge_weight, num_nodes=3)
    data = AddRemainingSelfLoops(attr='edge_weight', fill_value=5)(data)
    assert data.edge_index.tolist() == [[0, 1, 2, 0, 1, 2], [1, 2, 1, 0, 1, 2]]
    assert data.num_nodes == 3
    assert data.edge_weight.tolist() == [1, 3, 4, 2, 5, 5]

    data = Data(edge_index=edge_index, edge_attr=edge_attr, num_nodes=3)
    data = AddRemainingSelfLoops(attr='edge_attr', fill_value='add')(data)
    assert data.edge_index.tolist() == [[0, 1, 2, 0, 1, 2], [1, 2, 1, 0, 1, 2]]
    assert data.num_nodes == 3
    assert data.edge_attr.tolist() == [[1, 2], [5, 6], [7, 8], [3, 4], [8, 10],
                                       [5, 6]]


def test_add_remaining_self_loops_all_loops_exist():
    # All self-loops already exist in the data object.
    edge_index = torch.tensor([[0, 1, 2], [0, 1, 2]])
    data = Data(edge_index=edge_index, num_nodes=3)
    data = AddRemainingSelfLoops()(data)
    assert data.edge_index.tolist() == edge_index.tolist()

    # All self-loops already exist in the data object, some of them appear
    # multiple times.
    edge_index = torch.tensor([[0, 0, 1, 1, 2], [0, 0, 1, 1, 2]])
    data = Data(edge_index=edge_index, num_nodes=3)
    data = AddRemainingSelfLoops()(data)
    assert data.edge_index.tolist() == [[0, 1, 2], [0, 1, 2]]


def test_hetero_add_remaining_self_loops():
    edge_index = torch.tensor([[0, 0, 1, 2], [1, 0, 2, 1]])

    data = HeteroData()
    data['v'].num_nodes = 3
    data['w'].num_nodes = 3
    data['v', 'v'].edge_index = edge_index
    data['v', 'w'].edge_index = edge_index
    data = AddRemainingSelfLoops()(data)
    assert data['v', 'v'].edge_index.tolist() == [[0, 1, 2, 0, 1, 2],
                                                  [1, 2, 1, 0, 1, 2]]
    assert data['v', 'w'].edge_index.tolist() == edge_index.tolist()
