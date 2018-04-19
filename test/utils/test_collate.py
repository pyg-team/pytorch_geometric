import torch
from torch_geometric.utils import Data, collate_to_set, collate_to_batch


def test_collate_to_set():
    data1 = Data(x=torch.Tensor([1, 2, 3]), y=torch.Tensor([[0]]))
    data2 = Data(x=torch.Tensor([4, 5]), y=torch.Tensor([[1], [2]]))
    output, slices = collate_to_set([data1, data2])

    assert output.num_keys == 2
    assert output.x.tolist() == [1, 2, 3, 4, 5]
    assert output.y.tolist() == [[0], [1], [2]]
    assert slices['x'].tolist() == [0, 3, 5]
    assert slices['y'].tolist() == [0, 1, 3]


def test_collate_to_batch():
    data1 = Data(x=torch.Tensor([1, 2, 3]), y=torch.Tensor([[0]]))
    data2 = Data(x=torch.Tensor([4, 5]), y=torch.Tensor([[1], [2]]))
    output = collate_to_batch([data1, data2])

    assert output.num_keys == 3
    assert output.x.tolist() == [1, 2, 3, 4, 5]
    assert output.y.tolist() == [[0], [1], [2]]
    assert output.batch.tolist() == [0, 0, 0, 1, 1]
