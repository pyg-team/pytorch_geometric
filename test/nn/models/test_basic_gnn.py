import os
from itertools import product

import pytest
import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d, ReLU

from torch_geometric.nn import LayerNorm
from torch_geometric.nn.models import GAT, GCN, GIN, PNA, GraphSAGE

out_dims = [None, 8]
dropouts = [0.0, 0.5]
acts = [None, 'leaky_relu', torch.relu_, F.elu, ReLU()]
norms = [None, BatchNorm1d(16), LayerNorm(16)]
jks = ['last', 'cat', 'max', 'lstm']


@pytest.mark.parametrize('out_dim,dropout,act,norm,jk',
                         product(out_dims, dropouts, acts, norms, jks))
def test_gcn(out_dim, dropout, act, norm, jk):
    x = torch.randn(3, 8)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    out_channels = 16 if out_dim is None else out_dim

    model = GCN(8, 16, num_layers=2, out_channels=out_dim, dropout=dropout,
                act=act, norm=norm, jk=jk)
    assert str(model) == f'GCN(8, {out_channels}, num_layers=2)'
    assert model(x, edge_index).size() == (3, out_channels)


@pytest.mark.parametrize('out_dim,dropout,act,norm,jk',
                         product(out_dims, dropouts, acts, norms, jks))
def test_graph_sage(out_dim, dropout, act, norm, jk):
    x = torch.randn(3, 8)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    out_channels = 16 if out_dim is None else out_dim

    model = GraphSAGE(8, 16, num_layers=2, out_channels=out_dim,
                      dropout=dropout, act=act, norm=norm, jk=jk)
    assert str(model) == f'GraphSAGE(8, {out_channels}, num_layers=2)'
    assert model(x, edge_index).size() == (3, out_channels)


@pytest.mark.parametrize('out_dim,dropout,act,norm,jk',
                         product(out_dims, dropouts, acts, norms, jks))
def test_gin(out_dim, dropout, act, norm, jk):
    x = torch.randn(3, 8)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    out_channels = 16 if out_dim is None else out_dim

    model = GIN(8, 16, num_layers=2, out_channels=out_dim, dropout=dropout,
                act=act, norm=norm, jk=jk)
    assert str(model) == f'GIN(8, {out_channels}, num_layers=2)'
    assert model(x, edge_index).size() == (3, out_channels)


@pytest.mark.parametrize('out_dim,dropout,act,norm,jk',
                         product(out_dims, dropouts, acts, norms, jks))
def test_gat(out_dim, dropout, act, norm, jk):
    x = torch.randn(3, 8)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    out_channels = 16 if out_dim is None else out_dim

    model = GAT(8, 16, num_layers=2, out_channels=out_dim, dropout=dropout,
                act=act, norm=norm, jk=jk)
    assert str(model) == f'GAT(8, {out_channels}, num_layers=2)'
    assert model(x, edge_index).size() == (3, out_channels)

    model = GAT(8, 16, num_layers=2, out_channels=out_dim, dropout=dropout,
                act=act, norm=norm, jk=jk, heads=4)
    assert str(model) == f'GAT(8, {out_channels}, num_layers=2)'
    assert model(x, edge_index).size() == (3, out_channels)


@pytest.mark.parametrize('out_dim,dropout,act,norm,jk',
                         product(out_dims, dropouts, acts, norms, jks))
def test_pna(out_dim, dropout, act, norm, jk):
    x = torch.randn(3, 8)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    deg = torch.tensor([0, 2, 1])
    out_channels = 16 if out_dim is None else out_dim
    aggregators = ['mean', 'min', 'max', 'std', 'var', 'sum']
    scalers = [
        'identity', 'amplification', 'attenuation', 'linear', 'inverse_linear'
    ]

    model = PNA(8, 16, num_layers=2, out_channels=out_dim, dropout=dropout,
                act=act, norm=norm, jk=jk, aggregators=aggregators,
                scalers=scalers, deg=deg)
    assert str(model) == f'PNA(8, {out_channels}, num_layers=2)'
    assert model(x, edge_index).size() == (3, out_channels)


def test_packaging():
    x = torch.randn(3, 8)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])

    model = GraphSAGE(8, 16, num_layers=2)
    torch.save(model, 'model.pt')

    model = torch.load('model.pt')
    with torch.no_grad():
        assert model(x, edge_index).size() == (3, 16)
    os.remove('model.pt')

    model = GraphSAGE(8, 16, num_layers=2)
    with torch.package.PackageExporter('package.pt') as pe:
        pe.extern('torch_geometric.nn.**')
        pe.extern('_operator')
        pe.save_pickle('models', 'model.pkl', model)

    pi = torch.package.PackageImporter('package.pt')
    model = pi.load_pickle('models', 'model.pkl')
    with torch.no_grad():
        assert model(x, edge_index).size() == (3, 16)
    os.remove('package.pt')
