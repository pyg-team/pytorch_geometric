# import copy

import torch
from torch_sparse import SparseTensor
from torch_geometric.nn import MessagePassing


def test_message_passing_with_adj():
    row, col = torch.tensor([
        [0, 0, 0, 1, 1, 1],
        [0, 1, 2, 0, 1, 2],
    ])
    edge_index = torch.stack([row, col], dim=0)
    adj_t = SparseTensor(row=row, col=col).t()

    x = torch.Tensor([[1], [2]])

    conv = MessagePassing()

    out = conv.propagate(edge_index, x=x, size=(2, 3))
    print(out)
    out = conv.propagate(adj_t, x=x)
    print(out)


# def test_message_passing_with_edge_index():
#     edge_index = torch.tensor([
#         [0, 0, 0, 1, 1, 1],
#         [0, 1, 2, 0, 1, 2],
#     ])

#     x = torch.Tensor([[1], [2]])
#     out = MessagePassing(flow='source_to_target').propagate(
#         edge_index, x=x, size=(2, 3))
#     assert out.tolist() == [[3], [3], [3]]

#     x = torch.Tensor([[1], [2], [3]])
#     out = MessagePassing(flow='source_to_target').propagate(edge_index, x=x)
#     assert out.tolist() == [[3], [3], [3]]

#     x = torch.Tensor([[1], [2], [3]])
#     out = MessagePassing(flow='target_to_source').propagate(
#         edge_index, x=x, size=(2, 3))
#     assert out.tolist() == [[6], [6]]

#     x = torch.Tensor([[1], [2], [3]])
#     out = MessagePassing(flow='target_to_source').propagate(edge_index, x=x)
#     assert out.tolist() == [[6], [6], [0]]

#     x = (torch.Tensor([[1], [2]]), torch.Tensor([[1], [2], [3]]))
#     out = MessagePassing(flow='source_to_target').propagate(edge_index, x=x)
#     assert out.tolist() == [[3], [3], [3]]

#     x = (torch.Tensor([[1], [2]]), torch.Tensor([[1], [2], [3]]))
#     out = MessagePassing(flow='target_to_source').propagate(edge_index, x=x)
#     assert out.tolist() == [[6], [6]]

# class MyConv(MessagePassing):
#     def __init__(self):
#         super(MyConv, self).__init__(aggr='add')
#         self.weight = torch.nn.Parameter(torch.randn(16, 16))

#     def forward(self, x, edge_index, **kwargs):
#         return self.propagate(edge_index, x=x, **kwargs)

#     def message(self, x, x_j, edge_index, size, zeros=True):
#         assert size == [x.size(self.node_dim), x.size(self.node_dim)]
#         assert x_j.size(self.node_dim) == edge_index.size(1)
#         return x_j * 0 if zeros else x_j

# def test_default_arguments():
#     edge_index = torch.tensor([
#         [0, 0, 0, 1, 1, 1],
#         [0, 1, 2, 0, 1, 2],
#     ])
#     x = torch.Tensor([[1], [2], [3]])

#     conv = MyConv()
#     out = conv(x, edge_index)
#     assert (out == 0).all()
#     out = conv(x, edge_index, zeros=False)
#     assert (out != 0).all()

# def test_copy():
#     conv = MyConv()
#     conv2 = copy.copy(conv)

#     assert conv != conv2
#     assert conv.weight.data_ptr == conv2.weight.data_ptr

#     conv = copy.deepcopy(conv)
#     assert conv != conv2
#     assert conv.weight.data_ptr != conv2.weight.data_ptr
