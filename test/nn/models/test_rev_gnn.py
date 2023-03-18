import pytest
import torch

from torch_geometric.nn import GraphConv, GroupAddRev, SAGEConv
from torch_geometric.nn.dense.linear import Linear


@pytest.mark.parametrize('num_groups', [2, 4, 8, 16])
def test_revgnn_forward_inverse(num_groups):
    x = torch.randn(4, 32)
    edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])

    lin = Linear(32, 32)
    conv = SAGEConv(32 // num_groups, 32 // num_groups)
    conv = GroupAddRev(conv, num_groups=num_groups)
    assert str(conv) == (f'GroupAddRev(SAGEConv({32 // num_groups}, '
                         f'{32 // num_groups}, aggr=mean), '
                         f'num_groups={num_groups})')

    h = lin(x)
    h_o = h.clone().detach()

    out = conv(h, edge_index)
    assert h.untyped_storage().size() == 0

    h_rev = conv.inverse(out, edge_index)
    assert torch.allclose(h_o, h_rev, atol=0.001)


@pytest.mark.parametrize('num_groups', [2, 4, 8, 16])
def test_revgnn_backward(num_groups):
    x = torch.randn(4, 32)
    edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])

    lin = Linear(32, 32)
    conv = SAGEConv(32 // num_groups, 32 // num_groups)
    conv = GroupAddRev(conv, num_groups=num_groups)

    h = lin(x)
    out = conv(h, edge_index)
    target = out.mean()
    target.backward()


@pytest.mark.parametrize('num_groups', [2, 4, 8, 16])
def test_revgnn_multi_backward(num_groups):
    x = torch.randn(4, 32)
    edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])

    lin = Linear(32, 32)
    conv = SAGEConv(32 // num_groups, 32 // num_groups)
    conv = GroupAddRev(conv, num_groups=num_groups, num_bwd_passes=4)

    h = lin(x)
    out = conv(h, edge_index)
    target = out.mean()
    target.backward(retain_graph=True)
    target.backward(retain_graph=True)
    torch.autograd.grad(outputs=target, inputs=[h] + list(conv.parameters()),
                        retain_graph=True)
    torch.autograd.grad(outputs=target, inputs=[h] + list(conv.parameters()))


@pytest.mark.parametrize('num_groups', [2, 4, 8, 16])
def test_revgnn_diable(num_groups):
    x = torch.randn(4, 32)
    edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])

    lin = Linear(32, 32)
    conv = SAGEConv(32 // num_groups, 32 // num_groups)
    conv = GroupAddRev(conv, num_groups=num_groups, disable=True)

    h = lin(x)
    out = conv(h, edge_index)
    target = out.mean()
    target.backward()

    # Memory will not be freed if disable:
    assert h.untyped_storage().size() == 4 * 4 * 32


@pytest.mark.parametrize('num_groups', [2, 4, 8, 16])
def test_revgnn_with_args(num_groups):
    x = torch.randn(4, 32)
    edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    edge_weight = torch.rand(4)

    lin = Linear(32, 32)
    conv = GraphConv(32 // num_groups, 32 // num_groups)
    conv = GroupAddRev(conv, num_groups=num_groups)

    h = lin(x)
    out = conv(h, edge_index, edge_weight)
    target = out.mean()
    target.backward()
