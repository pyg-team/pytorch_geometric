from math import exp

import torch
from torch.autograd import Variable
from torch_geometric.utils import softmax


def test_softmax():
    row = torch.LongTensor([0, 0, 0, 1, 1, 1])
    col = torch.LongTensor([1, 2, 3, 0, 2, 3])
    edge_attr = torch.Tensor([0, 1, 2, 0, 1, 2])
    e = [exp(1), exp(2), exp(3)]
    e_sum = e[0] + e[1] + e[2]
    e = [e[0] / e_sum, e[1] / e_sum, e[2] / e_sum]

    output = softmax(edge_attr, row)

    output = softmax(Variable(edge_attr), row)

    output = softmax(edge_attr, col)

    edge_attr = torch.Tensor([[1, 1], [2, 2], [3, 3], [1, 1], [2, 2], [3, 3]])
    output = softmax(edge_attr, row)

    output = softmax(Variable(edge_attr), row)
    output
