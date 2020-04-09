import torch
from torch_sparse import SparseTensor
from torch_geometric.nn.conv.utils.message_passing import MessagePassing


class MyConv(MessagePassing):
    def __init__(self):
        super(MyConv, self).__init__(aggr='max')

    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:

        return self.propagate(edge_index, x=x, y=x)

    # def sparse_message_and_aggregate(self, adj_t, x_j):
    #     return adj_t.matmul(x_j, reduce=self.aggr)

    # def dense_message_and_aggregate(self, adj_t, x_j):
    #     return adj_t.matmul(x_j)

    # def partial_aggregate(self, inputs, inv_edge_mask):
    # print(inputs.size(), edge_mask.size())
    # inputs.masked_fill_(edge_mask, 0)
    # out = (inputs * edge_attr.unsqueeze(-1))
    # out = out.sum(dim=-2)
    # return inputs.sum(dim=-2)

    def message(self, x_j):
        return x_j


def test_message_passing():
    row = torch.tensor([0, 1, 1, 2, 2])
    col = torch.tensor([1, 0, 2, 0, 1])
    x = torch.Tensor([1, 2, 3]).view(-1, 1)

    sparse_adj_t = SparseTensor(row=row, col=col).t()
    dense_adj_t = sparse_adj_t.to_dense()
    edge_index = torch.stack([row, col], dim=0)

    conv = MyConv()
    print(conv)

    print(conv.supports_sparse_fused_format())
    print(conv.supports_dense_fused_format())
    print(conv.supports_sparse_format())
    print(conv.supports_partial_format())

    print(conv.get_adj_format(sparse_adj_t))
    print(conv.get_adj_format(dense_adj_t))
    print(conv.get_adj_format(edge_index))

    out1 = conv.propagate(edge_index, x=x)
    out2 = conv.propagate(sparse_adj_t, x=x)
    out3 = conv.propagate(dense_adj_t, x=x)
    assert out1.tolist() == out2.tolist()
    assert out1.tolist() == out3.tolist()

    conv.check_propagate_consistency(sparse_adj_t, x=x)

    # Static graph.
    # x = torch.Tensor([[1, 2, 3], [1, 2, 3]]).view(2, -1, 1)
    # out1 = conv.propagate(edge_index, x=x)
    # out2 = conv.propagate(sparse_adj_t, x=x)
    # out3 = conv.propagate(dense_adj_t.unsqueeze(0), x=x)
    # assert out1.tolist() == out2.tolist()
    # assert out2.tolist() == out3.tolist()
