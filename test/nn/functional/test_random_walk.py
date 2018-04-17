import torch
from torch.autograd import Variable as Var
from torch_geometric.nn.functional.random_walk import random_walk


def test_random_walk():
    edge_index = torch.LongTensor([[0, 0, 1, 1, 2], [0, 1, 1, 2, 2]])
    edge_attr = torch.Tensor([0.5, 0.5, 0.5, 0.5, 1])
    one_hot = torch.Tensor([[0, 1], [1, 0], [0, 1]])
    weight = torch.Tensor(2, 4).fill_(1)  # 2 classes, 4 steps.

    random_walk(edge_index, edge_attr, one_hot, weight)
    random_walk(edge_index, Var(edge_attr), Var(one_hot), Var(weight))
