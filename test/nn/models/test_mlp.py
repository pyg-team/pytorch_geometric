from itertools import product

import pytest
import torch

from torch_geometric.nn import MLP


@pytest.mark.parametrize('batch_norm,relu_first',
                         product([False, True], [False, True]))
def test_mlp(batch_norm, relu_first):
    x = torch.randn(4, 16)

    torch.manual_seed(12345)
    mlp = MLP([16, 32, 32, 64], batch_norm=batch_norm, relu_first=relu_first)
    assert str(mlp) == 'MLP(16, 32, 32, 64)'
    out = mlp(x)
    assert out.size() == (4, 64)

    jit = torch.jit.script(mlp)
    assert torch.allclose(jit(x), out)

    torch.manual_seed(12345)
    mlp = MLP(16, hidden_channels=32, out_channels=64, num_layers=3,
              batch_norm=batch_norm, relu_first=relu_first)
    assert torch.allclose(mlp(x), out)
