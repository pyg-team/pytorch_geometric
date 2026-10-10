import pytest
import torch
import torch.nn.functional as F

from torch_geometric.nn import EGNNConv
from torch_geometric.nn.conv.egnn_conv import egnn_norm
from torch_geometric.testing import withDevice, withPackage
from torch_geometric.typing import SparseTensor
from torch_geometric.utils import (
    to_torch_coo_tensor,
    to_torch_csc_tensor,
    to_torch_csr_tensor,
)


def to_dense(edge_index, edge_attr, num_nodes):
    # Rows are targets, columns are sources, as in the paper.
    adj = edge_attr.new_zeros(num_nodes, num_nodes, edge_attr.size(-1))
    return adj.index_put((edge_index[1], edge_index[0]), edge_attr,
                         accumulate=True)


def reference_norm(adj):
    # Independent dense implementation of equations (1)-(2).
    adj = adj + torch.diag_embed(
        (adj.sum(1) == 0).T.to(adj.dtype)).movedim(0, -1)
    row_normalized = adj / adj.sum(1, keepdim=True)
    columns = row_normalized.sum(0, keepdim=True)
    scaled = row_normalized / columns.masked_fill(columns == 0, 1.)
    return torch.einsum('ikp,jkp->ijp', scaled, row_normalized)


@withDevice
@pytest.mark.parametrize('edge_dim', [1, 3])
def test_egnn_norm(device, edge_dim):
    edge_index = torch.tensor([[0, 0, 1, 2, 2, 3], [0, 1, 2, 0, 3, 1]],
                              device=device)
    edge_attr = torch.rand(6, edge_dim, device=device, dtype=torch.double)
    index, attr = egnn_norm(edge_index, edge_attr, 5)
    actual = to_dense(index, attr, 5)
    expected = reference_norm(to_dense(edge_index, edge_attr, 5))
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        actual.sum(0),
        torch.ones(5, edge_dim, device=device, dtype=torch.double))
    torch.testing.assert_close(actual.sum(1), actual.sum(0))
    torch.testing.assert_close(actual, actual.transpose(0, 1))
    assert (actual >= 0).all()
    assert actual[0, 1].min() > 0  # Two targets share source node 0.
    assert actual[4, 4].min() == 1  # Isolated node.


@pytest.mark.parametrize('backbone', ['gcn', 'gat'])
@pytest.mark.parametrize('edge_dim', [1, 3])
@pytest.mark.parametrize('normalize', [False, True])
@withDevice
def test_egnn_conv(backbone, edge_dim, normalize, device):
    torch.manual_seed(12345)
    x = torch.randn(4, 8, device=device, dtype=torch.double)
    edge_index = torch.tensor([[0, 0, 1, 2, 2, 3], [0, 1, 2, 0, 3, 1]],
                              device=device)
    edge_attr = torch.rand(6, edge_dim, device=device, dtype=torch.double)
    conv = EGNNConv(8, 4, edge_dim, backbone=backbone, normalize=normalize,
                    add_self_loops=False).to(device)
    conv = conv.double()
    assert str(conv) == (f"EGNNConv(8, 4, edge_dim={edge_dim}, "
                         f"backbone='{backbone}')")
    out, (index, attr) = conv(x, edge_index, edge_attr,
                              return_edge_weights=True)
    assert out.shape == (4, edge_dim * 4)
    torch.testing.assert_close(conv(x, edge_index, edge_attr), out)
    torch.testing.assert_close(
        conv(x, edge_index, edge_attr, return_edge_weights=False), out)

    adj = to_dense(edge_index, edge_attr, 4)
    if normalize:
        adj = reference_norm(adj)
    h = F.linear(x, conv.lin.weight)
    if backbone == 'gat':
        # Equations (9)-(11), evaluated directly without sparse softmax.
        source = (h * conv.att_src).sum(-1)
        target = (h * conv.att_dst).sum(-1)
        attention = F.leaky_relu(target[:, None] + source[None, :], 0.2)
        adj = reference_norm(attention.exp().unsqueeze(-1) * adj)
    expected = torch.einsum('ijp,jf->ipf', adj, h).reshape(4, -1)
    torch.testing.assert_close(out, expected + conv.bias)
    torch.testing.assert_close(to_dense(index, attr, 4), adj)


@pytest.mark.parametrize('backbone', ['gcn', 'gat'])
@pytest.mark.parametrize('layout', ['coo', 'csr', 'csc'])
@pytest.mark.parametrize('edge_dim', [1, 3])
def test_egnn_conv_sparse(backbone, layout, edge_dim):
    x = torch.randn(4, 8)
    edge_index = torch.tensor([[0, 0, 1, 2, 3], [1, 2, 2, 0, 1]])
    edge_attr = torch.rand(5, edge_dim)
    make_sparse = {
        'coo': to_torch_coo_tensor,
        'csr': to_torch_csr_tensor,
        'csc': to_torch_csc_tensor,
    }[layout]
    # Construct the transposed adjacency directly, including dense channels.
    adj_t = make_sparse(edge_index.flip(0), edge_attr, size=(4, 4))
    conv = EGNNConv(8, 4, edge_dim, backbone=backbone)
    expected = conv(x, edge_index, edge_attr, return_edge_weights=True)
    actual = conv(x, adj_t, return_edge_weights=True)
    torch.testing.assert_close(actual, expected)
    with pytest.raises(ValueError, match='store'):
        conv(x, adj_t, edge_attr)
    with pytest.raises(ValueError, match='square'):
        conv(x[:-1], adj_t)


@withPackage('torch_sparse')
@pytest.mark.parametrize('backbone', ['gcn', 'gat'])
def test_egnn_conv_sparse_tensor(backbone):
    x = torch.randn(4, 8)
    edge_index = torch.tensor([[0, 0, 1, 2], [1, 2, 0, 3]])
    edge_attr = torch.rand(4, 3)
    adj_t = SparseTensor.from_edge_index(edge_index.flip(0), edge_attr,
                                         sparse_sizes=(4, 4))
    conv = EGNNConv(8, 4, 3, backbone=backbone)
    torch.testing.assert_close(conv(x, adj_t), conv(x, edge_index, edge_attr))


@pytest.mark.parametrize('backbone', ['gcn', 'gat'])
def test_egnn_conv_duplicates_and_uncoalesced(backbone):
    x = torch.randn(3, 4)
    edge_index = torch.tensor([[0, 0, 1, 2], [1, 1, 2, 0]])
    edge_attr = torch.rand(4, 2)
    adj_t = torch.sparse_coo_tensor(edge_index.flip(0), edge_attr, (3, 3, 2))
    assert not adj_t.is_coalesced()
    conv = EGNNConv(4, 3, 2, backbone=backbone, add_self_loops=False)
    torch.testing.assert_close(conv(x, adj_t), conv(x, edge_index, edge_attr))
    assert not adj_t.is_coalesced()  # No mutation of the caller's tensor.


@pytest.mark.parametrize('backbone', ['gcn', 'gat'])
@pytest.mark.parametrize('num_nodes', [0, 1, 4])
def test_egnn_conv_empty(backbone, num_nodes):
    x = torch.randn(num_nodes, 4, requires_grad=True)
    edge_index = torch.empty(2, 0, dtype=torch.long)
    edge_attr = torch.empty(0, 2)
    conv = EGNNConv(4, 3, 2, backbone=backbone, add_self_loops=False)
    out, (index, attr) = conv(x, edge_index, edge_attr,
                              return_edge_weights=True)
    expected = conv.lin(x).repeat(1, 2) + conv.bias
    torch.testing.assert_close(out, expected)
    torch.testing.assert_close(
        to_dense(index, attr, num_nodes),
        torch.eye(num_nodes).unsqueeze(-1).repeat(1, 1, 2))
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


@pytest.mark.parametrize('backbone', ['gcn', 'gat'])
@pytest.mark.parametrize('normalize', [False, True])
def test_egnn_conv_zero_channels(backbone, normalize):
    x = torch.randn(3, 4, requires_grad=True)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    edge_attr = torch.tensor([[1., 0.], [0., 0.], [2., 0.]],
                             requires_grad=True)
    conv = EGNNConv(4, 3, 2, backbone=backbone, normalize=normalize,
                    add_self_loops=False)
    out, (index, attr) = conv(x, edge_index, edge_attr,
                              return_edge_weights=True)
    assert torch.isfinite(out).all() and torch.isfinite(attr).all()
    out.square().sum().backward()
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(edge_attr.grad).all()
    if normalize or backbone == 'gat':
        adj = to_dense(index, attr, 3)
        torch.testing.assert_close(adj.sum(0), torch.ones(3, 2))
        torch.testing.assert_close(adj[:, :, 1], torch.eye(3))


@pytest.mark.parametrize('backbone', ['gcn', 'gat'])
def test_egnn_conv_gradients(backbone):
    torch.manual_seed(12345)
    x = torch.randn(3, 2, dtype=torch.double, requires_grad=True)
    edge_index = torch.tensor([[0, 0, 1, 1, 2, 2], [0, 1, 1, 2, 0, 2]])
    edge_attr = torch.rand(6, 2, dtype=torch.double, requires_grad=True)
    conv = EGNNConv(2, 2, 2, backbone=backbone, add_self_loops=False).double()
    params = dict(conv.named_parameters())

    def forward(x, edge_attr, *weights):
        return torch.func.functional_call(conv, dict(zip(params, weights)),
                                          (x, edge_index, edge_attr))

    assert torch.autograd.gradcheck(forward, (x, edge_attr, *params.values()))
    out, (_, attr) = conv(x, edge_index, edge_attr, return_edge_weights=True)
    (out.square().sum() + attr.square().sum()).backward()
    for value in [x, edge_attr, *params.values()]:
        assert value.grad is not None
        assert torch.isfinite(value.grad).all()
        assert value.grad.abs().sum() > 0


@pytest.mark.parametrize('backbone', ['gcn', 'gat'])
def test_egnn_conv_defaults_lazy_and_bias(backbone):
    x = torch.randn(4, 8)
    index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    conv = EGNNConv(-1, 4, 2, backbone=backbone, bias=False)
    out = conv(x, index)
    assert conv.bias is None
    assert conv.lin.weight.shape == (4, 8)
    torch.testing.assert_close(out, conv(x, index, torch.ones(3, 2)))
    conv.reset_parameters()
    assert torch.isfinite(conv(x, index)).all()
    single = EGNNConv(8, 4, 1, backbone=backbone)
    torch.testing.assert_close(single(x, index, torch.ones(3)),
                               single(x, index, torch.ones(3, 1)))


def test_egnn_conv_dropout_and_attention_stability():
    x = torch.randn(4, 2)
    index = torch.tensor([[0, 0, 1, 2, 3], [1, 2, 2, 0, 1]])
    conv = EGNNConv(2, 3, 2, dropout=1., bias=False)
    with torch.no_grad():
        conv.att_src.fill_(1000.)
        conv.att_dst.fill_(-1000.)
    out, (edge_index, attr) = conv(x, index, return_edge_weights=True)
    torch.testing.assert_close(out, torch.zeros(4, 6))
    assert torch.isfinite(attr).all()
    torch.testing.assert_close(
        to_dense(edge_index, attr, 4).sum(0), torch.ones(4, 2))
    conv.eval()
    out = conv(x, index)
    assert out.abs().sum() > 0 and torch.isfinite(out).all()
    torch.testing.assert_close(out, conv(x, index))


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
        'dropout': -0.1
    }, 'dropout'),
    ({
        'dropout': 1.1
    }, 'dropout'),
    ({
        'normalize': False
    }, 'self-loops'),
])
def test_egnn_conv_invalid_config(kwargs, match):
    args = dict(in_channels=4, out_channels=3, edge_dim=2)
    args.update(kwargs)
    with pytest.raises(ValueError, match=match):
        EGNNConv(**args)


@pytest.mark.parametrize('value', [-1., float('nan'), float('inf')])
def test_egnn_conv_invalid_edges(value):
    conv = EGNNConv(4, 3, 1)
    with pytest.raises(ValueError, match='finite and nonnegative'):
        conv(torch.randn(2, 4), torch.tensor([[0], [1]]),
             torch.tensor([value]))
    with pytest.raises(ValueError, match='shape'):
        conv(torch.randn(2, 4), torch.tensor([[0], [1]]), torch.ones(1, 2))
    with pytest.raises(ValueError, match='two-dimensional'):
        conv(torch.randn(1, 2, 4), torch.tensor([[0], [1]]))


@pytest.mark.parametrize('backbone', ['gcn', 'gat'])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_egnn_conv_low_precision(backbone, dtype):
    x = torch.randn(3, 4, dtype=dtype, requires_grad=True)
    index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    conv = EGNNConv(4, 2, 2, backbone=backbone).to(dtype)
    out = conv(x, index)
    assert out.dtype == dtype and torch.isfinite(out).all()
    out.float().square().sum().backward()
    assert torch.isfinite(x.grad).all()


@pytest.mark.parametrize('backbone', ['gcn', 'gat'])
def test_egnn_conv_self_loops_and_channel_order(backbone):
    x = torch.randn(3, 8)[:, ::2]
    index = torch.tensor([[0, 0, 0, 1, 2], [0, 0, 1, 2, 0]])
    attr = torch.rand(5, 3)
    conv = EGNNConv(4, 3, 3, backbone=backbone, bias=False)
    out = conv(x, index, attr)
    # Existing duplicate loops are summed and missing loops receive ones.
    loop_index = torch.tensor([[0, 0, 1, 1, 2, 2], [0, 1, 1, 2, 0, 2]])
    loop_attr = torch.stack([
        attr[0] + attr[1], attr[2],
        torch.ones(3), attr[3], attr[4],
        torch.ones(3)
    ])
    manual = EGNNConv(4, 3, 3, backbone=backbone, add_self_loops=False,
                      bias=False)
    manual.load_state_dict(conv.state_dict())
    torch.testing.assert_close(out, manual(x, loop_index, loop_attr))
    order = torch.tensor([2, 0, 1])
    permuted = conv(x, index, attr[:, order])
    torch.testing.assert_close(permuted.reshape(3, 3, 3),
                               out.reshape(3, 3, 3)[:, order])


@pytest.mark.parametrize('backbone', ['gcn', 'gat'])
@pytest.mark.parametrize('layout', ['coo', 'csr', 'csc'])
def test_egnn_conv_sparse_value_gradients(backbone, layout):
    x = torch.randn(3, 4, requires_grad=True)
    index = torch.tensor([[0, 0, 1, 2], [1, 2, 2, 0]])
    attr = torch.rand(4, 2, requires_grad=True)
    make_sparse = {
        'coo': to_torch_coo_tensor,
        'csr': to_torch_csr_tensor,
        'csc': to_torch_csc_tensor,
    }[layout]
    adj_t = make_sparse(index.flip(0), attr, size=(3, 3))
    conv = EGNNConv(4, 3, 2, backbone=backbone)
    sparse = conv(x, adj_t)
    sparse_grads = torch.autograd.grad(sparse.square().sum(), (x, attr))
    out = conv(x, index, attr)
    coo_grads = torch.autograd.grad(out.square().sum(), (x, attr))
    torch.testing.assert_close(sparse_grads, coo_grads)


@pytest.mark.parametrize('scale', [1e-38, 1e38])
def test_egnn_norm_scale_invariance(scale):
    index = torch.cartesian_prod(torch.arange(3), torch.arange(3)).T
    attr = torch.ones(9, 2)
    expected = egnn_norm(index, attr, 3)
    torch.testing.assert_close(egnn_norm(index, attr * scale, 3), expected)


@pytest.mark.parametrize('backbone', ['gcn', 'gat'])
def test_egnn_conv_scalar_sparse_values(backbone):
    x = torch.randn(3, 4)
    index = torch.tensor([[0, 0, 1, 2], [1, 2, 2, 0]])
    conv = EGNNConv(4, 3, 1, backbone=backbone)
    adj_t = to_torch_coo_tensor(index.flip(0), size=(3, 3))
    torch.testing.assert_close(conv(x, adj_t), conv(x, index))
