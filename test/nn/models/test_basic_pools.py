import pytest
from itertools import product

import torch
import torch.nn.functional as F
from torch.nn import ReLU, BatchNorm1d
from torch_geometric.nn import LayerNorm
from torch_geometric.nn.models import ASAPool, EdgePool, Graclus, SAGPool

dropouts = [0.0, 0.5]
acts = [None, torch.relu_, F.elu, ReLU()]
norms = [None, BatchNorm1d(16), LayerNorm(16)]
jks = ['last', 'cat', 'max', 'lstm']
glob_pools = ['mean', 'max', 'add']
ratios = [.75, .5]


@pytest.mark.parametrize('dropout,act,norm,jk,glob_pool,ratio',
                         product(dropouts, acts, norms, jks, glob_pools, ratios))
def test_asapool(dropout, act, norm, jk, glob_pool, ratio):
    x = torch.randn(3, 8)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    in_channels = 8
    hidden_channels = 16
    num_layers = 2
    out_channels = hidden_channels if jk != 'cat' else num_layers*hidden_channels

    model = ASAPool(in_channels=in_channels, hidden_channels=hidden_channels, num_layers=num_layers,
                dropout=dropout, act=act, norm=norm, jk=jk, glob_pool=glob_pool, ratio=ratio)
    assert str(model) == f'ASAPool(8, {out_channels}, num_layers=2)' #todo:change out_channels
    assert model(x, edge_index).size() == (1, out_channels)

@pytest.mark.parametrize('dropout,act,norm,jk,glob_pool',
                         product(dropouts, acts, norms, jks, glob_pools))
def test_edgepool(dropout, act, norm, jk, glob_pool):
    x = torch.randn(3, 8)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    in_channels = 8
    hidden_channels = 16
    num_layers = 2
    out_channels = hidden_channels if jk != 'cat' else num_layers*hidden_channels

    model = EdgePool(in_channels=in_channels, hidden_channels=hidden_channels, num_layers=num_layers,
                dropout=dropout, act=act, norm=norm, jk=jk, glob_pool=glob_pool)
    assert str(model) == f'EdgePool(8, {out_channels}, num_layers=2)' #todo:change out_channels
    assert model(x, edge_index).size() == (1, out_channels)

@pytest.mark.parametrize('dropout,act,norm,jk,glob_pool,ratio',
                         product(dropouts, acts, norms, jks, glob_pools, ratios))
def test_sagpool(dropout, act, norm, jk, glob_pool, ratio):
    x = torch.randn(3, 8)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    in_channels = 8
    hidden_channels = 16
    num_layers = 2
    out_channels = hidden_channels if jk != 'cat' else num_layers*hidden_channels

    model = SAGPool(in_channels=in_channels, hidden_channels=hidden_channels, num_layers=num_layers,
                dropout=dropout, act=act, norm=norm, jk=jk, glob_pool=glob_pool, ratio=ratio)
    assert str(model) == f'SAGPool(8, {out_channels}, num_layers=2)' #todo:change out_channels
    assert model(x, edge_index).size() == (1, out_channels)

@pytest.mark.parametrize('dropout,act,norm,jk,glob_pool',
                         product(dropouts, acts, norms, jks, glob_pools))
def test_graclus(dropout, act, norm, jk, glob_pool):
    x = torch.randn(3, 8)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    in_channels = 8
    hidden_channels = 16
    num_layers = 2
    out_channels = hidden_channels if jk != 'cat' else num_layers*hidden_channels

    model = Graclus(in_channels=in_channels, hidden_channels=hidden_channels, num_layers=num_layers,
                dropout=dropout, act=act, norm=norm, jk=jk, glob_pool=glob_pool)
    assert str(model) == f'Graclus(8, {out_channels}, num_layers=2)' #todo:change out_channels
    assert model(x, edge_index).size() == (1, out_channels)
