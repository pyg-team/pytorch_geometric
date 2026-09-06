import torch

from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import RemoveSelfLoops


def test_remove_self_loops():
    assert str(RemoveSelfLoops()) == 'RemoveSelfLoops()'

    assert len(RemoveSelfLoops()(Data())) == 0

    edge_index = torch.tensor([[0, 1, 1, 2], [0, 0, 1, 1]])
    edge_weight = torch.tensor([1, 2, 3, 4])
    edge_attr = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]])

    data = Data(edge_index=edge_index, num_nodes=3)
    data = RemoveSelfLoops()(data)
    assert len(data) == 2
    assert data.edge_index.tolist() == [[1, 2], [0, 1]]
    assert data.num_nodes == 3

    data = Data(edge_index=edge_index, edge_weight=edge_weight, num_nodes=3)
    data = RemoveSelfLoops(attr='edge_weight')(data)
    assert data.edge_index.tolist() == [[1, 2], [0, 1]]
    assert data.num_nodes == 3
    assert data.edge_weight.tolist() == [2, 4]

    data = Data(edge_index=edge_index, edge_attr=edge_attr, num_nodes=3)
    data = RemoveSelfLoops(attr='edge_attr')(data)
    assert data.edge_index.tolist() == [[1, 2], [0, 1]]
    assert data.num_nodes == 3
    assert data.edge_attr.tolist() == [[3, 4], [7, 8]]


def test_remove_self_loops_multiple_attrs():
    # A custom edge-level attribute (e.g. `time` in a temporal graph) should
    # stay in sync with `edge_index` when explicitly listed, and other
    # attributes not listed should be left untouched (and thus mismatched in
    # length with the filtered `edge_index`, since the caller opted out of
    # keeping them aligned).
    edge_index = torch.tensor([[0, 1, 1, 2], [0, 0, 1, 1]])
    edge_weight = torch.tensor([1, 2, 3, 4])
    time = torch.tensor([10, 20, 30, 40])

    data = Data(edge_index=edge_index, edge_weight=edge_weight, time=time,
                num_nodes=3)
    data = RemoveSelfLoops(attr=['edge_weight', 'time'])(data)
    assert data.edge_index.tolist() == [[1, 2], [0, 1]]
    assert data.edge_weight.tolist() == [2, 4]
    assert data.time.tolist() == [20, 40]

    # `time` alone, without touching `edge_weight`:
    data = Data(edge_index=edge_index, edge_weight=edge_weight, time=time,
                num_nodes=3)
    data = RemoveSelfLoops(attr=['time'])(data)
    assert data.edge_index.tolist() == [[1, 2], [0, 1]]
    assert data.time.tolist() == [20, 40]
    assert data.edge_weight.tolist() == edge_weight.tolist()  # untouched


def test_hetero_remove_self_loops():
    edge_index = torch.tensor([[0, 1, 1, 2], [0, 0, 1, 1]])

    data = HeteroData()
    data['v'].num_nodes = 3
    data['w'].num_nodes = 3
    data['v', 'v'].edge_index = edge_index
    data['v', 'w'].edge_index = edge_index
    data = RemoveSelfLoops()(data)
    assert data['v', 'v'].edge_index.tolist() == [[1, 2], [0, 1]]
    assert data['v', 'w'].edge_index.tolist() == edge_index.tolist()
