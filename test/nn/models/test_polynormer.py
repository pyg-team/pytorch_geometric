import pytest
import torch

from torch_geometric.nn.models import Polynormer


@pytest.mark.parametrize('qkv_bias', [True, False])
@pytest.mark.parametrize('pre_ln', [True, False])
@pytest.mark.parametrize('post_bn', [True, False])
def test_polynormer(qkv_bias, pre_ln, post_bn):
    x = torch.randn(10, 16)
    edge_index = torch.tensor([
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    ])
    batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

    model = Polynormer(
        in_channels=16,
        hidden_channels=128,
        out_channels=40,
        qkv_bias=qkv_bias,
        pre_ln=pre_ln,
        post_bn=post_bn,
    )
    out = model(x, edge_index, batch)
    assert out.size() == (10, 40)
    model._global = True
    out = model(x, edge_index, batch)
    assert out.size() == (10, 40)
