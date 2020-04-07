import torch
from torch_geometric.nn.conv.message_passing2 import MessagePassing


class MyConv(MessagePassing):
    def __init__(self):
        super(MyConv, self).__init__()

    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:

        return self.propagate(edge_index, x=x, y=x)

    def message(self, x_j):
        return x_j


def test_message_passing():
    edge_index = torch.tensor([[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]])
    x = torch.tensor([[1.], [2.], [3.]])

    conv = MyConv()
    # out = conv(x, edge_index)  # Example input

    # traced_bla = torch.jit.script(bla)
    # print(traced_bla)

    # conv = torch.jit.script(conv)  # We can now convert it to a ScriptModule.
    # out = conv(x, edge_index)
    # print(out.size())
