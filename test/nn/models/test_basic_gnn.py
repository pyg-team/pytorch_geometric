import os
import os.path as osp
import warnings
from itertools import product

import pytest
import torch
import torch.nn.functional as F

import torch_geometric.typing
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv
from torch_geometric.nn.models import GAT, GCN, GIN, PNA, EdgeCNN, GraphSAGE
from torch_geometric.testing import withPackage

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


def test_packaging():
    warnings.filterwarnings('ignore', '.*TypedStorage is deprecated.*')

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


@withPackage('onnx', 'onnxruntime')
def test_onnx(tmp_path, capfd):
    import onnx
    import onnxruntime as ort

    warnings.filterwarnings('ignore', '.*tensor to a Python boolean.*')
    warnings.filterwarnings('ignore', '.*shape inference of prim::Constant.*')

    class MyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = SAGEConv(8, 16)
            self.conv2 = SAGEConv(16, 16)

        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index).relu()
            x = self.conv2(x, edge_index)
            return x

    model = MyModel()
    x = torch.randn(3, 8)
    edge_index = torch.tensor([[0, 1, 2], [1, 0, 2]])
    expected = model(x, edge_index)
    assert expected.size() == (3, 16)

    path = osp.join(tmp_path, 'model.onnx')
    torch.onnx.export(model, (x, edge_index), path,
                      input_names=('x', 'edge_index'), opset_version=16)
    if torch_geometric.typing.WITH_PT2:
        out, _ = capfd.readouterr()
        assert '0 NONE 0 NOTE 0 WARNING 0 ERROR' in out

    model = onnx.load(path)
    onnx.checker.check_model(model)

    providers = ['CPUExecutionProvider']
    ort_session = ort.InferenceSession(path, providers=providers)

    out = ort_session.run(None, {
        'x': x.numpy(),
        'edge_index': edge_index.numpy()
    })[0]
    out = torch.from_numpy(out)
    assert torch.allclose(out, expected, atol=1e-6)
