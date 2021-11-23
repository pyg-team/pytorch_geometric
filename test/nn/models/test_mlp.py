import pytest
from itertools import product

import torch
from torch_geometric.nn import MLP


@pytest.mark.parametrize('batch_norm,relu_first',
                         product([False, True], [False, True]))
def test_mlp(batch_norm, relu_first):
    x = torch.randn(4, 16)

    mlp = MLP([16, 32, 32, 64], batch_norm=batch_norm, relu_first=relu_first)
    assert mlp.__repr__() == 'MLP(16, 32, 32, 64)'
    out = mlp(x)
    assert out.size() == (4, 64)

    jit = torch.jit.script(mlp)
    assert torch.allclose(jit(x), out)
