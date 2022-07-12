from itertools import product

import pytest
import torch

from torch_geometric.nn import MLP
from torch_geometric.testing import is_full_test


@pytest.mark.parametrize(
    'norm, act_first, plain_last',
    product(['batch_norm', None], [False, True], [False, True]),
)
def test_mlp(norm, act_first, plain_last):
    x = torch.randn(4, 16)

    torch.manual_seed(12345)
    mlp = MLP(
        [16, 32, 32, 64],
        norm=norm,
        act_first=act_first,
        plain_last=plain_last,
    )
    assert str(mlp) == 'MLP(16, 32, 32, 64)'
    out = mlp(x)
    assert out.size() == (4, 64)

    if is_full_test():
        jit = torch.jit.script(mlp)
        assert torch.allclose(jit(x), out)

    torch.manual_seed(12345)
    mlp = MLP(
        16,
        hidden_channels=32,
        out_channels=64,
        num_layers=3,
        norm=norm,
        act_first=act_first,
        plain_last=plain_last,
    )
    assert torch.allclose(mlp(x), out)
