import io

import pytest
import torch
import torch.nn.functional as F

from torch_geometric.data import Batch, Data
from torch_geometric.nn import EGNN, global_max_pool
from torch_geometric.nn.conv.egnn_conv import egnn_norm
from torch_geometric.utils import add_remaining_self_loops, to_torch_coo_tensor


@pytest.mark.parametrize('backbone', ['gcn', 'gat'])
@pytest.mark.parametrize('num_layers', [1, 2, 3])
@pytest.mark.parametrize('act', ['elu', None, torch.nn.ReLU()])
def test_egnn(backbone, num_layers, act):
    x = torch.randn(4, 5)
    index = torch.tensor([[0, 0, 1, 2, 3], [1, 2, 2, 0, 1]])
    attr = torch.rand(5, 3)
    model = EGNN(5, 8, 7, num_layers, 3, backbone=backbone, act=act)
    assert str(model) == (f"EGNN(5, 7, num_layers={num_layers}, "
                          f"backbone='{backbone}')")
    out, (edge_index, edge_attr) = model(x, index, attr,
                                         return_edge_weights=True)
    assert out.shape == (4, 7)
    torch.testing.assert_close(out, model(x, index, attr))
    adj_t = to_torch_coo_tensor(index.flip(0), attr, size=(4, 4))
    torch.testing.assert_close(out, model(x, adj_t))

    # Explicit layer-by-layer reference checks edge evolution and ordering.
    h, edges, weights = x, index, attr
    first_edges = None
    for i, conv in enumerate(model.convs):
        h, (edges, weights) = conv(h, edges, weights, return_edge_weights=True)
        if i == 0:
            first_edges = (edges, weights)
        if i < num_layers - 1 and model.act is not None:
            h = model.act(h)
    torch.testing.assert_close(out, model.lin(h))
    torch.testing.assert_close((edge_index, edge_attr), (edges, weights))
    if num_layers > 1:
        if backbone == 'gcn':
            torch.testing.assert_close((edges, weights), first_edges)
        else:
            assert weights.shape != first_edges[1].shape or not torch.allclose(
                weights, first_edges[1])
    model.reset_parameters()
    assert torch.isfinite(model(x, index, attr)).all()


@pytest.mark.parametrize('backbone', ['gcn', 'gat'])
def test_egnn_batch_permutation_and_graph_prediction(backbone):
    torch.manual_seed(12345)
    index = torch.tensor([[0, 0, 1, 2], [1, 2, 2, 0]])
    data1 = Data(x=torch.randn(3, 4), edge_index=index,
                 edge_attr=torch.rand(4, 2))
    data2 = Data(x=torch.randn(2, 4), edge_index=torch.tensor([[0], [1]]),
                 edge_attr=torch.rand(1, 2))
    batch = Batch.from_data_list([data1, data2])
    model = EGNN(4, 3, 6, 2, 2, backbone=backbone)
    out, (edges, attr) = model(batch.x, batch.edge_index, batch.edge_attr,
                               return_edge_weights=True)
    separate = torch.cat(
        [model(d.x, d.edge_index, d.edge_attr) for d in [data1, data2]])
    torch.testing.assert_close(out, separate)
    assert (batch.batch[edges[0]] == batch.batch[edges[1]]).all()

    permutation = torch.randperm(5)
    inverse = permutation.argsort()
    permuted = model(batch.x[permutation], inverse[batch.edge_index],
                     batch.edge_attr)
    torch.testing.assert_close(permuted, out[permutation])
    head = torch.nn.Linear(6, 2)
    predictions = head(global_max_pool(out, batch.batch))
    assert predictions.shape == (2, 2)
    F.cross_entropy(predictions, torch.tensor([0, 1])).backward()
    for value in model.parameters():
        assert value.grad is not None and torch.isfinite(value.grad).all()
        assert value.grad.abs().sum() > 0
    assert attr.requires_grad == (backbone == 'gat')


@pytest.mark.parametrize('backbone', ['gcn', 'gat'])
def test_egnn_edge_gradients_and_serialization(backbone):
    x = torch.randn(4, 3, requires_grad=True)
    index = torch.tensor([[0, 0, 1, 2, 3], [1, 2, 2, 0, 1]])
    attr = torch.rand(5, 2, requires_grad=True)
    model = EGNN(-1, 4, 5, 3, 2, backbone=backbone)
    out = model(x, index, attr)
    out.square().sum().backward()
    assert x.grad.abs().sum() > 0 and attr.grad.abs().sum() > 0
    assert torch.isfinite(x.grad).all() and torch.isfinite(attr.grad).all()
    stream = io.BytesIO()
    torch.save(model.state_dict(), stream)
    stream.seek(0)
    copy = EGNN(3, 4, 5, 3, 2, backbone=backbone)
    copy.load_state_dict(torch.load(stream, weights_only=True))
    torch.testing.assert_close(copy(x, index, attr), out)


@pytest.mark.parametrize('backbone', ['gcn', 'gat'])
def test_egnn_dropout_train_eval(backbone):
    x = torch.randn(4, 3)
    index = torch.tensor([[0, 0, 1, 2, 3], [1, 2, 2, 0, 1]])
    model = EGNN(3, 4, 5, 2, 2, backbone=backbone, dropout=0.5)
    torch.manual_seed(1)
    first = model(x, index)
    torch.manual_seed(2)
    assert not torch.allclose(first, model(x, index))
    model.eval()
    torch.testing.assert_close(model(x, index), model(x, index))


@pytest.mark.parametrize('backbone', ['gcn', 'gat'])
def test_egnn_pre_normalized_input(backbone):
    torch.manual_seed(12345)
    x = torch.randn(4, 3, dtype=torch.double)
    index = torch.tensor([[0, 0, 1, 2, 3], [1, 2, 2, 0, 1]])
    attr = torch.rand(5, 2, dtype=torch.double)
    loops, loop_attr = add_remaining_self_loops(index, attr, fill_value=1.,
                                                num_nodes=4)
    edges, weights = egnn_norm(loops, loop_attr, 4)
    raw = EGNN(3, 4, 5, 2, 2, backbone=backbone).double()
    prepared = EGNN(3, 4, 5, 2, 2, backbone=backbone, normalize=False).double()
    prepared.load_state_dict(raw.state_dict())
    assert raw.normalize and not prepared.normalize
    assert all(not conv.normalize and not conv.add_self_loops
               for conv in prepared.convs)
    expected = raw(x, index, attr, return_edge_weights=True)
    actual = prepared(x, edges, weights, return_edge_weights=True)
    torch.testing.assert_close(actual, expected)
    actual[0].square().sum().backward()
    expected[0].square().sum().backward()
    for param1, param2 in zip(prepared.parameters(), raw.parameters()):
        torch.testing.assert_close(param1.grad, param2.grad)


@pytest.mark.parametrize('kwargs,match', [
    ({
        'num_layers': 0
    }, 'num_layers'),
    ({
        'out_channels': 0
    }, 'out_channels'),
    ({
        'normalize': False,
        'add_self_loops': True,
    }, 'self-loops'),
    ({
        'backbone': 'sage'
    }, 'backbone'),
])
def test_egnn_invalid_config(kwargs, match):
    args = dict(in_channels=4, hidden_channels=3, out_channels=2, num_layers=2,
                edge_dim=2)
    args.update(kwargs)
    with pytest.raises(ValueError, match=match):
        EGNN(**args)
