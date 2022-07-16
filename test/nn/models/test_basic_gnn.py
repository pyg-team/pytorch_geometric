import os
import os.path as osp
from itertools import product

import pytest
import torch
import torch.nn.functional as F

from torch_geometric.loader import NeighborLoader
from torch_geometric.nn.models import GAT, GCN, GIN, PNA, EdgeCNN, GraphSAGE
from torch_geometric.testing import withPython

out_dims = [None, 8]
dropouts = [0.0, 0.5]
acts = [None, 'leaky_relu', torch.relu_, F.elu, torch.nn.ReLU()]
norms = [None, 'batch_norm', 'layer_norm']
jks = [None, 'last', 'cat', 'max', 'lstm']


@pytest.mark.parametrize('out_dim,dropout,act,norm,jk',
                         product(out_dims, dropouts, acts, norms, jks))
def test_gcn(out_dim, dropout, act, norm, jk):
    x = torch.randn(3, 8)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    out_channels = 16 if out_dim is None else out_dim

    model = GCN(8, 16, num_layers=3, out_channels=out_dim, dropout=dropout,
                act=act, norm=norm, jk=jk)
    assert str(model) == f'GCN(8, {out_channels}, num_layers=3)'
    assert model(x, edge_index).size() == (3, out_channels)


@pytest.mark.parametrize('out_dim,dropout,act,norm,jk',
                         product(out_dims, dropouts, acts, norms, jks))
def test_graph_sage(out_dim, dropout, act, norm, jk):
    x = torch.randn(3, 8)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    out_channels = 16 if out_dim is None else out_dim

    model = GraphSAGE(8, 16, num_layers=3, out_channels=out_dim,
                      dropout=dropout, act=act, norm=norm, jk=jk)
    assert str(model) == f'GraphSAGE(8, {out_channels}, num_layers=3)'
    assert model(x, edge_index).size() == (3, out_channels)


@pytest.mark.parametrize('out_dim,dropout,act,norm,jk',
                         product(out_dims, dropouts, acts, norms, jks))
def test_gin(out_dim, dropout, act, norm, jk):
    x = torch.randn(3, 8)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    out_channels = 16 if out_dim is None else out_dim

    model = GIN(8, 16, num_layers=3, out_channels=out_dim, dropout=dropout,
                act=act, norm=norm, jk=jk)
    assert str(model) == f'GIN(8, {out_channels}, num_layers=3)'
    assert model(x, edge_index).size() == (3, out_channels)


@pytest.mark.parametrize('out_dim,dropout,act,norm,jk',
                         product(out_dims, dropouts, acts, norms, jks))
def test_gat(out_dim, dropout, act, norm, jk):
    x = torch.randn(3, 8)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    out_channels = 16 if out_dim is None else out_dim

    for v2 in [False, True]:
        model = GAT(8, 16, num_layers=3, out_channels=out_dim, v2=v2,
                    dropout=dropout, act=act, norm=norm, jk=jk)
        assert str(model) == f'GAT(8, {out_channels}, num_layers=3)'
        assert model(x, edge_index).size() == (3, out_channels)

        model = GAT(8, 16, num_layers=3, out_channels=out_dim, v2=v2,
                    dropout=dropout, act=act, norm=norm, jk=jk, heads=4)
        assert str(model) == f'GAT(8, {out_channels}, num_layers=3)'
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

    model = PNA(8, 16, num_layers=3, out_channels=out_dim, dropout=dropout,
                act=act, norm=norm, jk=jk, aggregators=aggregators,
                scalers=scalers, deg=deg)
    assert str(model) == f'PNA(8, {out_channels}, num_layers=3)'
    assert model(x, edge_index).size() == (3, out_channels)


@pytest.mark.parametrize('out_dim,dropout,act,norm,jk',
                         product(out_dims, dropouts, acts, norms, jks))
def test_edge_cnn(out_dim, dropout, act, norm, jk):
    x = torch.randn(3, 8)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    out_channels = 16 if out_dim is None else out_dim

    model = EdgeCNN(8, 16, num_layers=3, out_channels=out_dim, dropout=dropout,
                    act=act, norm=norm, jk=jk)
    assert str(model) == f'EdgeCNN(8, {out_channels}, num_layers=3)'
    assert model(x, edge_index).size() == (3, out_channels)


@pytest.mark.parametrize('out_dim,jk', product(out_dims, jks))
def test_one_layer_gnn(out_dim, jk):
    x = torch.randn(3, 8)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    out_channels = 16 if out_dim is None else out_dim

    model = GraphSAGE(8, 16, num_layers=1, out_channels=out_dim, jk=jk)
    assert model(x, edge_index).size() == (3, out_channels)


@pytest.mark.parametrize('jk', [None, 'last'])
def test_basic_gnn_inference(get_dataset, jk):
    dataset = get_dataset(name='Cora')
    data = dataset[0]

    model = GraphSAGE(dataset.num_features, hidden_channels=16, num_layers=2,
                      out_channels=dataset.num_classes, jk=jk)
    model.eval()

    out1 = model(data.x, data.edge_index)
    assert out1.size() == (data.num_nodes, dataset.num_classes)

    loader = NeighborLoader(data, num_neighbors=[-1], batch_size=128)
    out2 = model.inference(loader)
    assert out1.size() == out2.size()
    assert torch.allclose(out1, out2, atol=1e-4)

    assert 'n_id' not in data


@withPython('3.7', '3.8', '3.9')  # Packaging does not support Python 3.10 yet.
def test_packaging():
    os.makedirs(torch.hub._get_torch_home(), exist_ok=True)

    x = torch.randn(3, 8)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])

    model = GraphSAGE(8, 16, num_layers=3)
    path = osp.join(torch.hub._get_torch_home(), 'pyg_test_model.pt')
    torch.save(model, path)

    model = torch.load(path)
    with torch.no_grad():
        assert model(x, edge_index).size() == (3, 16)

    model = GraphSAGE(8, 16, num_layers=3)
    path = osp.join(torch.hub._get_torch_home(), 'pyg_test_package.pt')
    with torch.package.PackageExporter(path) as pe:
        pe.extern('torch_geometric.nn.**')
        pe.extern('_operator')
        pe.save_pickle('models', 'model.pkl', model)

    pi = torch.package.PackageImporter(path)
    model = pi.load_pickle('models', 'model.pkl')
    with torch.no_grad():
        assert model(x, edge_index).size() == (3, 16)
