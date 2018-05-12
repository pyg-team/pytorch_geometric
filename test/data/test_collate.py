import torch
from torch_geometric.data import Data, collate_to_set, collate_to_batch

x1 = torch.tensor([1, 2, 3], dtype=torch.float)
edge_index1 = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
x2 = torch.tensor([1, 2], dtype=torch.float)
edge_index2 = torch.tensor([[0, 1], [1, 0]])
data1, data2 = Data(x1, edge_index1), Data(x2, edge_index2)


def test_collate_to_set():
    data, slices = collate_to_set([data1, data2])

    assert len(data) == 2
    assert data.x.tolist() == [1, 2, 3, 1, 2]
    assert data.edge_index.tolist() == [[0, 1, 1, 2, 0, 1], [1, 0, 2, 1, 1, 0]]
    assert len(slices.keys()) == 2
    assert slices['x'].tolist() == [0, 3, 5]
    assert slices['edge_index'].tolist() == [0, 4, 6]


def test_collate_to_batch():
    data = collate_to_batch([data1, data2])

    assert len(data) == 3
    assert data.x.tolist() == [1, 2, 3, 1, 2]
    assert data.edge_index.tolist() == [[0, 1, 1, 2, 3, 4], [1, 0, 2, 1, 4, 3]]
    assert data.batch.tolist() == [0, 0, 0, 1, 1]
