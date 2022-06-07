import pytest
import torch

import torch_geometric
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


@pytest.mark.parametrize('aggr_tuple', [
    (torch_geometric.nn.aggr.MeanAggregation, 'mean'),
    (torch_geometric.nn.aggr.SumAggregation, 'sum'),
    (torch_geometric.nn.aggr.MaxAggregation, 'max'),
    (torch_geometric.nn.aggr.MinAggregation, 'min'),
    (torch_geometric.nn.aggr.MulAggregation, 'mul'),
    (torch_geometric.nn.aggr.VarAggregation, 'var'),
    (torch_geometric.nn.aggr.StdAggregation, 'std'),
    (torch_geometric.nn.aggr.SoftmaxAggregation, 'softmax'),
    (torch_geometric.nn.aggr.PowerMeanAggregation, 'powermean'),
])
def test_aggregation_resolver(aggr_tuple):
    aggr_module, aggr_repr = aggr_tuple
    assert isinstance(aggregation_resolver(aggr_module()), aggr_module)
    assert isinstance(aggregation_resolver(aggr_repr), aggr_module)
