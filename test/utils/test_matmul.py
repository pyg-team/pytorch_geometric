import torch
from torch.autograd import Variable
from torch_geometric.utils import matmul


def test_matmul():
    edge_index = torch.LongTensor([[0, 0, 1, 2, 2], [0, 2, 1, 0, 1]])
    edge_attr = torch.Tensor([1, 2, 4, 1, 3])
    a = torch.sparse.FloatTensor(edge_index, edge_attr, torch.Size([3, 3]))
    b = torch.Tensor([[1, 4], [2, 5], [3, 6]])
    expected_output = torch.matmul(a.to_dense(), b).tolist()

    output = matmul(edge_index, edge_attr, b)
    assert output.tolist() == expected_output

    output = matmul(edge_index, Variable(edge_attr), Variable(b))
    assert output.data.tolist() == expected_output
