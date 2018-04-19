from torch import Tensor, LongTensor
from torch_geometric.data import Data, data_from_set, split_set


def test_data_from_set():
    dataset = Data(x=Tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
    slices = {'x': LongTensor([0, 3, 5, 8, 10])}
    output = data_from_set(dataset, slices, 1)

    assert len(output) == 1
    assert output.x.tolist() == [3, 4]


def test_split_set():
    dataset = Data(x=Tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
    slices = {'x': LongTensor([0, 3, 5, 8, 10])}
    split = LongTensor([3, 0, 2])
    output, output_slices = split_set(dataset, slices, split)

    assert len(output) == 1
    assert output.x.tolist() == [8, 9, 0, 1, 2, 5, 6, 7]
    assert len(slices.keys()) == 1
    assert output_slices['x'].tolist() == [0, 2, 5, 8]
