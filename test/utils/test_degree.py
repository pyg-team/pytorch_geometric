import torch
from torch.autograd import Variable
from torch_geometric.utils import degree


def test_degree():
    index = torch.LongTensor([0, 1, 0, 2, 0])
    expected_output = [3, 1, 1]

    output = degree(index)
    assert output.tolist() == expected_output

    output = degree(index, out=Variable(torch.FloatTensor()))
    assert output.data.tolist() == expected_output
