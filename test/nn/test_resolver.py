import pytest
import torch

from torch_geometric.nn import (
    LSTMAggregation,
    MaxAggregation,
    MeanAggregation,
    MinAggregation,
    MulAggregation,
    PowerMeanAggregation,
    SoftmaxAggregation,
    StdAggregation,
    SumAggregation,
    VarAggregation,
)
from torch_geometric.nn.resolver import (
    activation_resolver,
    aggregation_resolver,
)


def test_activation_resolver():
    assert isinstance(activation_resolver(torch.nn.ELU()), torch.nn.ELU)
    assert isinstance(activation_resolver(torch.nn.ReLU()), torch.nn.ReLU)
    assert isinstance(activation_resolver(torch.nn.PReLU()), torch.nn.PReLU)

    assert isinstance(activation_resolver('elu'), torch.nn.ELU)
    assert isinstance(activation_resolver('relu'), torch.nn.ReLU)
    assert isinstance(activation_resolver('prelu'), torch.nn.PReLU)


@pytest.mark.parametrize('aggr_test', [
    (MeanAggregation, 'mean'),
    (SumAggregation, 'sum'),
    (MaxAggregation, 'max'),
    (MinAggregation, 'min'),
    (MulAggregation, 'mul'),
    (VarAggregation, 'var'),
    (StdAggregation, 'std'),
    (SoftmaxAggregation, 'softmax'),
    (PowerMeanAggregation, 'powermean'),
])
def test_aggregation_resolver_basic(aggr_test):
    aggr_module, aggr_str = aggr_test
    assert isinstance(aggregation_resolver(aggr_module()), aggr_module)
    assert isinstance(aggregation_resolver(aggr_str), aggr_module)


@pytest.mark.parametrize('aggr_test', [
    (SoftmaxAggregation, 'softmax', {
        'learn': True
    }),
    (PowerMeanAggregation, 'powermean', {
        'learn': True
    }),
    (LSTMAggregation, 'lstm', {
        'in_channels': 16,
        'out_channels': 32
    }),
])
def test_aggregation_resolver_learnable(aggr_test):
    aggr_module, aggr_str, kwargs = aggr_test
    aggr = aggr_module(**kwargs)
    res_aggr1 = aggregation_resolver(aggr_module(**kwargs))
    res_aggr2 = aggregation_resolver(aggr_str, **kwargs)
    assert isinstance(res_aggr1, aggr_module)
    assert isinstance(res_aggr2, aggr_module)
    if issubclass(aggr_module, SoftmaxAggregation):
        repr = 'SoftmaxAggregation(learn=True)'
    elif issubclass(aggr_module, PowerMeanAggregation):
        repr = 'PowerMeanAggregation(learn=True)'
    elif issubclass(aggr_module, LSTMAggregation):
        repr = 'LSTMAggregation(16, 32)'
    else:
        raise ValueError("Unknown Aggregation")

    assert str(aggr) == repr
    assert str(res_aggr1) == repr
    assert str(res_aggr2) == repr
