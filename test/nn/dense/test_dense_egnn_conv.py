import pytest
import torch

from torch_geometric.nn import DenseEGNNConv, EGNNConv
from torch_geometric.testing import withDevice


@withDevice
@pytest.mark.parametrize('backbone', ['gcn', 'gat'])
@pytest.mark.parametrize('normalize', [False, True])
@pytest.mark.parametrize('edge_dim', [1, 3])
def test_dense_egnn_conv(device, backbone, normalize, edge_dim):
    torch.manual_seed(12345)
    x = torch.randn(5, 4, dtype=torch.double, device=device)
    # Two disconnected graphs with asymmetric, multichannel edge features.
    index = torch.tensor([[0, 0, 1, 2, 3, 4], [1, 2, 2, 0, 4, 3]],
                         device=device)
    attr = torch.rand(6, edge_dim, dtype=torch.double, device=device)
    sparse = EGNNConv(4, 3, edge_dim, backbone=backbone, normalize=normalize,
                      add_self_loops=normalize).to(device).double()
    dense = DenseEGNNConv(4, 3, edge_dim, backbone=backbone,
                          normalize=normalize,
                          add_self_loops=normalize).to(device).double()
    dense.load_state_dict(sparse.state_dict())
    assert str(dense) == (f"DenseEGNNConv(4, 3, edge_dim={edge_dim}, "
                          f"backbone='{backbone}')")

    adj = attr.new_zeros(2, 3, 3, edge_dim)
    adj[0, index[1, :4], index[0, :4]] = attr[:4]
    adj[1, index[1, 4:] - 3, index[0, 4:] - 3] = attr[4:]
    x_batch = torch.cat([x, x.new_zeros(1, 4)]).reshape(2, 3, 4)
    mask = torch.tensor([[True, True, True], [True, True, False]],
                        device=device)
    out, edges = dense(x_batch, adj, mask, return_edge_weights=True)
    expected, (edge_index, edge_attr) = sparse(x, index, attr,
                                               return_edge_weights=True)
    torch.testing.assert_close(out.reshape(6, -1)[:5], expected)
    torch.testing.assert_close(out, dense(x_batch, adj, mask))
    assert out[1, 2].abs().sum() == 0
    assert edges[1, 2].abs().sum() == edges[1, :, 2].abs().sum() == 0
    actual = attr.new_zeros(5, 5, edge_dim)
    actual[:3, :3] = edges[0]
    actual[3:, 3:] = edges[1, :2, :2]
    torch.testing.assert_close(actual[edge_index[1], edge_index[0]], edge_attr)


@pytest.mark.parametrize('backbone', ['gcn', 'gat'])
def test_dense_egnn_conv_broadcasting_and_mask(backbone):
    x = torch.randn(4, 3, 4)
    adj = torch.rand(3, 3, 2)
    conv = DenseEGNNConv(4, 3, 2, backbone=backbone)
    out = conv(x, adj)
    assert out.shape == (4, 3, 6)
    for i in range(4):
        torch.testing.assert_close(out[i:i + 1], conv(x[i], adj))
    torch.testing.assert_close(conv(x[0], adj.repeat(4, 1, 1, 1)),
                               conv(x[0].repeat(4, 1, 1), adj))
    mask = torch.tensor([True, False, True])
    out, edges = conv(x, adj, mask, return_edge_weights=True)
    assert out[:, 1].abs().sum() == 0
    assert edges[:, 1].abs().sum() == edges[:, :, 1].abs().sum() == 0
    assert torch.isfinite(out).all()
    torch.testing.assert_close(
        edges.sum(2),
        mask.to(out.dtype).reshape(1, 3, 1).expand(4, 3, 2))


@pytest.mark.parametrize('backbone', ['gcn', 'gat'])
@pytest.mark.parametrize('mask', [None, torch.tensor([[True, False, False]])])
def test_dense_egnn_conv_empty_channels_and_gradients(backbone, mask):
    x = torch.randn(1, 3, 2, dtype=torch.double, requires_grad=True)
    adj = torch.zeros(1, 3, 3, 2, dtype=torch.double, requires_grad=True)
    conv = DenseEGNNConv(2, 2, 2, backbone=backbone).double()
    out, edges = conv(x, adj, mask, return_edge_weights=True)
    assert torch.isfinite(out).all() and torch.isfinite(edges).all()
    out.square().sum().backward()
    assert torch.isfinite(x.grad).all() and torch.isfinite(adj.grad).all()
    strict = DenseEGNNConv(2, 2, 2, backbone=backbone,
                           add_self_loops=False).double()
    with pytest.raises(ValueError, match='positive mass'):
        strict(x, adj, mask)


@pytest.mark.parametrize('backbone', ['gcn', 'gat'])
def test_dense_egnn_conv_partial_zero_loop(backbone):
    adj = torch.eye(3).unsqueeze(-1).repeat(1, 1, 2)
    adj[0, 0, 1] = 0
    conv = DenseEGNNConv(4, 3, 2, backbone=backbone)
    with pytest.raises(ValueError, match='positive mass'):
        conv(torch.randn(3, 4), adj)


@withDevice
@pytest.mark.parametrize('backbone', ['gcn', 'gat'])
@pytest.mark.parametrize('layout', ['coo', 'csr', 'csc'])
def test_dense_egnn_conv_zero_loops(device, backbone, layout):
    x = torch.randn(3, 4, device=device, dtype=torch.double)
    index = torch.tensor([[0, 1, 2, 0, 1, 2], [0, 1, 2, 1, 2, 0]],
                         device=device)
    attr = torch.rand(6, 2, device=device, dtype=torch.double)
    attr[:3] = 0
    adj = attr.new_zeros(3, 3, 2)
    adj[index[1], index[0]] = attr
    adj_t = torch.sparse_coo_tensor(index.flip(0), attr, (3, 3, 2)).coalesce()
    adj_t = adj_t.to_sparse(layout=getattr(torch, f'sparse_{layout}'))
    sparse = EGNNConv(4, 3, 2, backbone=backbone).to(device).double()
    dense = DenseEGNNConv(4, 3, 2, backbone=backbone).to(device).double()
    dense.load_state_dict(sparse.state_dict())
    expected, edges = dense(x, adj, return_edge_weights=True)
    for edge_index, edge_attr in [(index, attr), (adj_t, None)]:
        out, (indices, values) = sparse(x, edge_index, edge_attr,
                                        return_edge_weights=True)
        torch.testing.assert_close(out, expected.squeeze(0))
        torch.testing.assert_close(values, edges[0, indices[1], indices[0]])


@pytest.mark.parametrize('backbone', ['gcn', 'gat'])
def test_dense_egnn_conv_gradcheck(backbone):
    x = torch.randn(1, 3, 2, dtype=torch.double, requires_grad=True)
    adj = torch.rand(1, 3, 3, 2, dtype=torch.double, requires_grad=True)
    conv = DenseEGNNConv(2, 2, 2, backbone=backbone,
                         add_self_loops=False).double()
    assert torch.autograd.gradcheck(conv, (x, adj))


def test_dense_egnn_conv_dropout():
    x, adj = torch.randn(2, 3, 4), torch.rand(3, 3, 2)
    conv = DenseEGNNConv(4, 3, 2, dropout=1., bias=False)
    out, edges = conv(x, adj, return_edge_weights=True)
    torch.testing.assert_close(out, torch.zeros(2, 3, 6))
    torch.testing.assert_close(edges.sum(2), torch.ones(2, 3, 2))
    conv.eval()
    assert conv(x, adj).abs().sum() > 0


@pytest.mark.parametrize('value', [-1., float('nan'), float('inf')])
def test_dense_egnn_conv_invalid_edges(value):
    conv = DenseEGNNConv(4, 3, 1)
    with pytest.raises(ValueError, match='finite and nonnegative'):
        conv(torch.randn(2, 4), torch.full((2, 2, 1), value))
    with pytest.raises(ValueError, match='square'):
        conv(torch.randn(2, 4), torch.ones(2, 3, 1))
    with pytest.raises(ValueError, match='shape'):
        conv(torch.randn(2, 4), torch.ones(2, 2))
    with pytest.raises(ValueError, match='boolean'):
        conv(torch.randn(2, 4), torch.ones(2, 2, 1), torch.ones(2))


@pytest.mark.parametrize('backbone', ['gcn', 'gat'])
def test_dense_egnn_conv_empty_graph(backbone):
    conv = DenseEGNNConv(4, 3, 2, backbone=backbone)
    out, adj = conv(torch.empty(0, 4), torch.empty(0, 0, 2),
                    return_edge_weights=True)
    assert out.shape == (1, 0, 6) and adj.shape == (1, 0, 0, 2)
    out, adj = conv(torch.randn(2, 3, 4), torch.rand(2, 3, 3, 2),
                    mask=torch.zeros(2, 3, dtype=torch.bool),
                    return_edge_weights=True)
    assert out.abs().sum() == adj.abs().sum() == 0


@pytest.mark.parametrize('backbone', ['gcn', 'gat'])
@pytest.mark.parametrize('x_batch,adj_batch', [(0, 1), (1, 0), (0, 0)])
def test_dense_egnn_conv_empty_batch(backbone, x_batch, adj_batch):
    x = torch.randn(x_batch, 3, 4, requires_grad=True)
    adj = torch.ones(adj_batch, 3, 3, 2, requires_grad=True)
    mask = torch.empty(0, 3, dtype=torch.bool)
    conv = DenseEGNNConv(4, 3, 2, backbone=backbone)
    out, weights = conv(x, adj, mask, return_edge_weights=True)
    assert out.shape == (0, 3, 6) and weights.shape == (0, 3, 3, 2)
    out.sum().backward()
    assert x.grad is not None and adj.grad is not None
    assert x.grad.abs().sum() == adj.grad.abs().sum() == 0


@pytest.mark.parametrize('kwargs,match', [
    ({
        'backbone': 'sage'
    }, 'backbone'),
    ({
        'edge_dim': 0
    }, 'positive'),
    ({
        'out_channels': 0
    }, 'positive'),
    ({
        'dropout': 1.1
    }, 'dropout'),
    ({
        'normalize': False
    }, 'self-loops'),
])
def test_dense_egnn_conv_invalid_config(kwargs, match):
    args = dict(in_channels=4, out_channels=3, edge_dim=2)
    args.update(kwargs)
    with pytest.raises(ValueError, match=match):
        DenseEGNNConv(**args)


@pytest.mark.parametrize('scale', [1e-38, 1e38])
def test_dense_egnn_conv_scale_invariance(scale):
    x = torch.randn(3, 4)
    adj = torch.ones(3, 3, 2)
    conv = DenseEGNNConv(4, 3, 2, add_self_loops=False)
    torch.testing.assert_close(conv(x, adj * scale), conv(x, adj))


@pytest.mark.parametrize('backbone', ['gcn', 'gat'])
def test_dense_egnn_conv_lazy_and_bias(backbone):
    conv = DenseEGNNConv(-1, 3, 2, backbone=backbone, bias=False)
    x, adj = torch.randn(3, 4), torch.rand(3, 3, 2)
    assert conv(x, adj).shape == (1, 3, 6)
    assert conv.bias is None and conv.lin.weight.shape == (3, 4)
    conv.reset_parameters()
    assert torch.isfinite(conv(x, adj)).all()
