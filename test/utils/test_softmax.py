from math import exp

import torch
from torch.autograd import Variable
from torch_geometric.utils import softmax


def test_softmax():
    edge_index = torch.LongTensor([[0, 0, 0, 1, 1, 1], [1, 2, 3, 0, 2, 3]])
    edge_attr = torch.Tensor([0, 1, 2, 0, 1, 2])
    e = [exp(1), exp(2), exp(3)]
    e_sum = e[0] + e[1] + e[2]
    e = [e[0] / e_sum, e[1] / e_sum, e[2] / e_sum]

    row, col = edge_index
    output = softmax(row, edge_attr)

    output = softmax(row, Variable(edge_attr))

    output = softmax(col, edge_attr)

    edge_attr = torch.Tensor([[1, 1], [2, 2], [3, 3], [1, 1], [2, 2], [3, 3]])
    output = softmax(row, edge_attr)

    output = softmax(row, Variable(edge_attr))
    output
