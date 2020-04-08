import torch
from torch_sparse import SparseTensor
from torch_geometric.nn.conv.utils.message_passing import MessagePassing


class MyConv(MessagePassing):
    def __init__(self):
        super(MyConv, self).__init__()

    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:

        return self.propagate(edge_index, x=x, y=x)

    def message(self, x_j):
        return x_j


def test_message_passing():
    row = torch.tensor([0, 0, 1, 1, 2, 2])
    col = torch.tensor([1, 2, 0, 2, 0, 1])
    x = torch.Tensor([1, 2, 3])

    sparse_adj_t = SparseTensor(row=row, col=col).t()
    dense_adj_t = sparse_adj_t.to_dense()
    edge_index = torch.stack([row, col], dim=0)

    conv = MyConv()
    print(conv)

    print(conv.supports_fused_format())
    print(conv.supports_sparse_format())
    print(conv.supports_partial_format())

    print(conv.get_adj_format(sparse_adj_t))
    print(conv.get_adj_format(dense_adj_t))
    print(conv.get_adj_format(edge_index))

    out = conv.propagate(edge_index, x=x)
    print(out)
    # out = conv(x, edge_index)  # Example input

    # traced_bla = torch.jit.script(bla)
    # print(traced_bla)

    # conv = torch.jit.script(conv)  # We can now convert it to a ScriptModule.
    # out = conv(x, edge_index)
    # print(out.size())
