import pytest
import torch

from torch_geometric.nn.aggr import EquilibriumAggregation


@pytest.mark.parametrize('iter', [0, 1, 5])
@pytest.mark.parametrize('alpha', [0, .1, 5])
def test_equilibrium(iter, alpha):
    batch_size = 10
    feature_channels = 3
    output_channels = 2
    x = torch.randn(batch_size, feature_channels)
    model = EquilibriumAggregation(feature_channels, output_channels,
                                   num_layers=[10, 10], grad_iter=iter)

    assert str(model) == 'EquilibriumAggregation()'
    out = model(x)
    assert out.size() == (1, 2)

    out = model(x, dim_size=3)
    assert out.size() == (3, 2)
    assert torch.all(out[1:, :] == 0)


@pytest.mark.parametrize('iter', [0, 1, 5])
@pytest.mark.parametrize('alpha', [0, .1, 5])
def test_equilibrium_batch(iter, alpha):
    batch_1, batch_2 = 4, 6
    feature_channels = 3
    output_channels = 2
    x = torch.randn(batch_1 + batch_2, feature_channels)
    batch = torch.tensor([0 for _ in range(batch_1)] +
                         [1 for _ in range(batch_2)])

    model = EquilibriumAggregation(feature_channels, output_channels,
                                   num_layers=[10, 10], grad_iter=iter)

    assert str(model) == 'EquilibriumAggregation()'
    out = model(x, batch)
    assert out.size() == (2, 2)

    out = model(x, dim_size=3)
    assert out.size() == (3, 2)
    assert torch.all(out[1:, :] == 0)
