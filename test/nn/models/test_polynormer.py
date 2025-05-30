import pytest
import torch

from torch_geometric.nn.models import Polynormer


@pytest.mark.parametrize('local_attn', [True, False])
@pytest.mark.parametrize('qk_shared', [True, False])
@pytest.mark.parametrize('pre_ln', [True, False])
@pytest.mark.parametrize('post_bn', [True, False])
def test_polynormer(local_attn, qk_shared, pre_ln, post_bn):
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
        qk_shared=qk_shared,
        pre_ln=pre_ln,
        post_bn=post_bn,
        local_attn=local_attn,
    )
    out = model(x, edge_index, batch)
    assert out.size() == (10, 40)
    model._global = True
    out = model(x, edge_index, batch)
    assert out.size() == (10, 40)
