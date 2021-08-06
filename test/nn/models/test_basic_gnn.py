import pytest
from itertools import product

import torch
import torch.nn.functional as F
from torch.nn import ReLU, BatchNorm1d
from torch_geometric.nn import LayerNorm
from torch_geometric.nn.models import GCN, GraphSAGE, GIN, GAT

dropouts = [0.0, 0.5]
acts = [None, torch.relu_, F.elu, ReLU()]
norms = [None, BatchNorm1d(16), LayerNorm(16)]
jks = ['last', 'cat', 'max', 'lstm']


@pytest.mark.parametrize('dropout,act,norm,jk',
                         product(dropouts, acts, norms, jks))
def test_gcn(dropout, act, norm, jk):
    x = torch.randn(3, 8)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    out_channels = 16 if jk != 'cat' else 32

    model = GCN(in_channels=8, hidden_channels=16, num_layers=2,
                dropout=dropout, act=act, norm=norm, jk=jk)
    assert str(model) == f'GCN(8, {out_channels}, num_layers=2)'
    assert model(x, edge_index).size() == (3, out_channels)


@pytest.mark.parametrize('dropout,act,norm,jk',
                         product(dropouts, acts, norms, jks))
def test_graph_sage(dropout, act, norm, jk):
    x = torch.randn(3, 8)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    out_channels = 16 if jk != 'cat' else 32

    model = GraphSAGE(in_channels=8, hidden_channels=16, num_layers=2,
                      dropout=dropout, act=act, norm=norm, jk=jk)
    assert str(model) == f'GraphSAGE(8, {out_channels}, num_layers=2)'
    assert model(x, edge_index).size() == (3, out_channels)


@pytest.mark.parametrize('dropout,act,norm,jk',
                         product(dropouts, acts, norms, jks))
def test_gin(dropout, act, norm, jk):
    x = torch.randn(3, 8)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    out_channels = 16 if jk != 'cat' else 32

    model = GIN(in_channels=8, hidden_channels=16, num_layers=2,
                dropout=dropout, act=act, norm=norm, jk=jk)
    assert str(model) == f'GIN(8, {out_channels}, num_layers=2)'
    assert model(x, edge_index).size() == (3, out_channels)


@pytest.mark.parametrize('dropout,act,norm,jk',
                         product(dropouts, acts, norms, jks))
def test_gat(dropout, act, norm, jk):
    x = torch.randn(3, 8)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    out_channels = 16 if jk != 'cat' else 32

    model = GAT(in_channels=8, hidden_channels=16, num_layers=2,
                dropout=dropout, act=act, norm=norm, jk=jk)
    assert str(model) == f'GAT(8, {out_channels}, num_layers=2)'
    assert model(x, edge_index).size() == (3, out_channels)

    model = GAT(in_channels=8, hidden_channels=16, num_layers=2,
                dropout=dropout, act=act, norm=norm, jk=jk, heads=4)
    assert str(model) == f'GAT(8, {out_channels}, num_layers=2)'
    assert model(x, edge_index).size() == (3, out_channels)
