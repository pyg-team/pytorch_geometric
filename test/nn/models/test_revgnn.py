import pytest
import torch

from torch_geometric.nn import GroupAddRev, SAGEConv
from torch_geometric.nn.dense.linear import Linear

hidden_channels = 32


@pytest.mark.parametrize("num_groups", [2, 4, 8, 16])
def test_revgnn_forward_inverse(num_groups):
    x = torch.randn(4, hidden_channels)
    edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    seed_gnn = SAGEConv(
        hidden_channels // num_groups,
        hidden_channels // num_groups,
        aggr="mean",
    )
    gnn = GroupAddRev(seed_gnn, num_groups=num_groups)
    assert str(gnn) == f"GroupAddRev(num_groups={num_groups}, \
            disable=False"

    lin = Linear(hidden_channels, hidden_channels)

    # Test forward and inverse
    h = lin(x)
    h_o = h.clone().detach()
    out = gnn(h, edge_index)
    # Check if memory is freed
    assert h.storage().size() == 0
    h_rev = gnn.inverse(out, edge_index)
    # Check if recomputed hidden features are close
    # the orginal hidden features
    assert torch.allclose(h_o, h_rev, atol=0.001)


@pytest.mark.parametrize("num_groups", [2, 4, 8, 16])
def test_revgnn_backward(num_groups):
    x = torch.randn(4, hidden_channels)
    edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    seed_gnn = SAGEConv(
        hidden_channels // num_groups,
        hidden_channels // num_groups,
        aggr="mean",
    )
    gnn = GroupAddRev(seed_gnn, num_groups=num_groups)
    lin = Linear(hidden_channels, hidden_channels)
    # Test backward
    gnn = GroupAddRev(seed_gnn, num_groups=num_groups)
    h = lin(x)
    out = gnn(h, edge_index)
    target = out.mean()
    target.backward()


@pytest.mark.parametrize("num_groups", [2, 4, 8, 16])
def test_revgnn_multi_backward(num_groups):
    x = torch.randn(4, hidden_channels)
    edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    seed_gnn = SAGEConv(
        hidden_channels // num_groups,
        hidden_channels // num_groups,
        aggr="mean",
    )
    # Test multiple backward
    gnn = GroupAddRev(seed_gnn, num_groups=num_groups, num_bwd_passes=4)
    lin = Linear(hidden_channels, hidden_channels)
    h = lin(x)
    out = gnn(h, edge_index)
    target = out.mean()
    target.backward(retain_graph=True)
    target.backward(retain_graph=True)
    torch.autograd.grad(outputs=target, inputs=[h] + list(gnn.parameters()),
                        retain_graph=True)
    torch.autograd.grad(outputs=target, inputs=[h] + list(gnn.parameters()))


@pytest.mark.parametrize("num_groups", [2, 4, 8, 16])
def test_revgnn_diable(num_groups):
    x = torch.randn(4, hidden_channels)
    edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    seed_gnn = SAGEConv(
        hidden_channels // num_groups,
        hidden_channels // num_groups,
        aggr="mean",
    )
    gnn = GroupAddRev(seed_gnn, num_groups=num_groups)
    lin = Linear(hidden_channels, hidden_channels)
    gnn = GroupAddRev(seed_gnn, num_groups=num_groups, disable=True)
    # Test disable
    gnn.disable = True
    h = lin(x)
    out = gnn(h, edge_index)
    target = out.mean()
    target.backward()
    # Memory will not be freed if disable
    assert h.storage().size() == 4 * hidden_channels
