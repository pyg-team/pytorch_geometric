from torch import Tensor, LongTensor
from torch_geometric.data import Data, collate_to_set, collate_to_batch


def test_collate_to_set():
    data1 = Data(x=Tensor([1, 2, 3]), edge_index=LongTensor([[0, 1], [1, 2]]))
    data2 = Data(x=Tensor([4, 5]), edge_index=LongTensor([[0, 1], [1, 0]]))
    output, output_slices = collate_to_set([data1, data2])

    assert len(output) == 2
    assert output.x.tolist() == [1, 2, 3, 4, 5]
    assert output.edge_index.tolist() == [[0, 1], [1, 2], [0, 1], [1, 0]]
    assert len(output_slices.keys()) == 2
    assert output_slices['x'].tolist() == [0, 3, 5]
    assert output_slices['edge_index'].tolist() == [0, 2, 4]


def test_collate_to_batch():
    data1 = Data(x=Tensor([1, 2, 3]), edge_index=Tensor([[0, 1], [1, 2]]))
    data2 = Data(x=Tensor([4, 5]), edge_index=Tensor([[0, 1], [1, 0]]))
    output = collate_to_batch([data1, data2])

    assert len(output) == 3
    assert output.x.tolist() == [1, 2, 3, 4, 5]
    assert output.edge_index.tolist() == [[0, 1, 0, 1], [1, 2, 1, 0]]
    assert output.batch.tolist() == [0, 0, 0, 1, 1]
