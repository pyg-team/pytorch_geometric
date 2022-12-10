import pytest
import torch

from torch_geometric.nn import GCNConv
from torch_geometric.utils import intermediate_mp_output


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


@pytest.mark.parametrize('i', [-3, -2, -1, 0, 1, 2, 4])
def test_intermediate_mp_out(i):
    x = torch.randn(6, 5)
    edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]])

    model = GNNTest(5)
    out, actual_intermediate_outs = model(x, edge_index)
    if i == 4:
        with pytest.raises(ValueError, match="'i' has to take"):
            _ = intermediate_mp_output(i, model, x=x, edge_index=edge_index)
    else:
        intermediate_out = intermediate_mp_output(i, model, x=x,
                                                  edge_index=edge_index)
        index = i if i >= 0 else 3 + i
        assert torch.allclose(intermediate_out,
                              actual_intermediate_outs[index])
