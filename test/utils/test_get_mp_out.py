import pytest
import torch

from torch_geometric.nn import GCNConv
from torch_geometric.utils import get_message_passing_embeddings


class GNNTest(torch.nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 5)
        self.conv2 = GCNConv(5, 6)
        self.conv3 = GCNConv(6, 7)
        self.lin = torch.nn.Linear(7, 8)

    def forward(self, x, edge_index):
        x1 = self.conv1(x, edge_index)
        x2 = self.conv2(x1, edge_index)
        x3 = self.conv3(x2, edge_index)
        out = self.lin(x3)
        return out, [x1, x2, x3]


def test_get_intermediate_messagepassing_embeddings():
    x = torch.randn(6, 5)
    edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]])

    model = GNNTest(5)
    out, actual_intermediate_outs = model(x, edge_index)

    with pytest.warns(
            UserWarning,
            match="'model' does not have any 'MessagePassing' layers."):
        intermediate_outs = get_message_passing_embeddings(
            torch.nn.Linear(5, 5), x)
    assert len(intermediate_outs) == 0

    intermediate_outs = get_message_passing_embeddings(model, x=x,
                                                       edge_index=edge_index)
    assert len(intermediate_outs) == 3
    for expected, out in zip(intermediate_outs, actual_intermediate_outs):
        assert torch.allclose(expected, out)
