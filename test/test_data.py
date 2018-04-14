import torch
from torch_geometric.datasets import Data


def test_data():
    data = Data(torch.Tensor([1, 2, 3]), torch.LongTensor([1, 1, 1, 1, 1]))
    data.to_variable('x')
    data.to_tensor()
