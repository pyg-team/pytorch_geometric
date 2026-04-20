import pytest
import torch

from torch_geometric.contrib.nn.conv import MixupConv
from torch_geometric.nn import GCNConv


@pytest.mark.parametrize('conv_layer', [GCNConv])
def test_mixup_conv(conv_layer):
    in_channels, out_channels = 3, 16
    x1 = torch.randn(8, in_channels)
    x2 = torch.randn(8, in_channels)
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6, 7]])

    conv = MixupConv(in_channels, out_channels, conv_layer)
    out = conv(x1, edge_index, x2)

    assert out.size() == (8, out_channels)
