import pytest
import torch
from torch.optim.lr_scheduler import ConstantLR, LambdaLR, ReduceLROnPlateau

import torch_geometric
from torch_geometric.nn.resolver import (
    activation_resolver,
    aggregation_resolver,
    lr_scheduler_resolver,
    normalization_resolver,
    optimizer_resolver,
)


def test_activation_resolver():
    assert isinstance(activation_resolver(torch.nn.ELU()), torch.nn.ELU)
    assert isinstance(activation_resolver(torch.nn.ReLU()), torch.nn.ReLU)
    assert isinstance(activation_resolver(torch.nn.PReLU()), torch.nn.PReLU)

    assert isinstance(activation_resolver('elu'), torch.nn.ELU)
    assert isinstance(activation_resolver('relu'), torch.nn.ReLU)
    assert isinstance(activation_resolver('prelu'), torch.nn.PReLU)


@pytest.mark.parametrize('aggr_tuple', [
    (torch_geometric.nn.MeanAggregation, 'mean'),
    (torch_geometric.nn.SumAggregation, 'sum'),
    (torch_geometric.nn.SumAggregation, 'add'),
    (torch_geometric.nn.MaxAggregation, 'max'),
    (torch_geometric.nn.MinAggregation, 'min'),
    (torch_geometric.nn.MulAggregation, 'mul'),
    (torch_geometric.nn.VarAggregation, 'var'),
    (torch_geometric.nn.StdAggregation, 'std'),
    (torch_geometric.nn.SoftmaxAggregation, 'softmax'),
    (torch_geometric.nn.PowerMeanAggregation, 'powermean'),
])
def test_aggregation_resolver(aggr_tuple):
    aggr_module, aggr_repr = aggr_tuple
    assert isinstance(aggregation_resolver(aggr_module()), aggr_module)
    assert isinstance(aggregation_resolver(aggr_repr), aggr_module)


def test_multi_aggregation_resolver():
    aggr = aggregation_resolver(None)
    assert aggr is None

    aggr = aggregation_resolver(['sum', 'mean', None])
    assert isinstance(aggr, torch_geometric.nn.MultiAggregation)
    assert len(aggr.aggrs) == 3
    assert isinstance(aggr.aggrs[0], torch_geometric.nn.SumAggregation)
    assert isinstance(aggr.aggrs[1], torch_geometric.nn.MeanAggregation)
    assert aggr.aggrs[2] is None


@pytest.mark.parametrize('norm_tuple', [
    (torch_geometric.nn.BatchNorm, 'batch', (16, )),
    (torch_geometric.nn.BatchNorm, 'batch_norm', (16, )),
    (torch_geometric.nn.InstanceNorm, 'instance_norm', (16, )),
    (torch_geometric.nn.LayerNorm, 'layer_norm', (16, )),
    (torch_geometric.nn.GraphNorm, 'graph_norm', (16, )),
    (torch_geometric.nn.GraphSizeNorm, 'graphsize_norm', ()),
    (torch_geometric.nn.PairNorm, 'pair_norm', ()),
    (torch_geometric.nn.MessageNorm, 'message_norm', ()),
    (torch_geometric.nn.DiffGroupNorm, 'diffgroup_norm', (16, 4)),
])
def test_normalization_resolver(norm_tuple):
    norm_module, norm_repr, norm_args = norm_tuple
    assert isinstance(normalization_resolver(norm_module(*norm_args)),
                      norm_module)
    assert isinstance(normalization_resolver(norm_repr, *norm_args),
                      norm_module)


def test_optimizer_resolver():
    params = [torch.nn.Parameter(torch.Tensor(1))]

    assert isinstance(optimizer_resolver(torch.optim.SGD(params, lr=0.01)),
                      torch.optim.SGD)
    assert isinstance(optimizer_resolver(torch.optim.Adam(params)),
                      torch.optim.Adam)
    assert isinstance(optimizer_resolver(torch.optim.Rprop(params)),
                      torch.optim.Rprop)

    assert isinstance(optimizer_resolver('sgd', params, lr=0.01),
                      torch.optim.SGD)
    assert isinstance(optimizer_resolver('adam', params), torch.optim.Adam)
    assert isinstance(optimizer_resolver('rprop', params), torch.optim.Rprop)


@pytest.mark.parametrize('scheduler_args', [
    ('constant_with_warmup', LambdaLR),
    ('linear_with_warmup', LambdaLR),
    ('cosine_with_warmup', LambdaLR),
    ('cosine_with_warmup_restarts', LambdaLR),
    ('polynomial_with_warmup', LambdaLR),
    ('constant', ConstantLR),
    ('ReduceLROnPlateau', ReduceLROnPlateau),
])
def test_lr_scheduler_resolver(scheduler_args):
    scheduler_name, scheduler_cls = scheduler_args

    model = torch.nn.Linear(10, 5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    lr_scheduler = lr_scheduler_resolver(
        scheduler_name,
        optimizer,
        num_training_steps=100,
    )
    assert isinstance(lr_scheduler, scheduler_cls)
