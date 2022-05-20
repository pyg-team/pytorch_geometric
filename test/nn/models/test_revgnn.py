import pytest
import torch

from torch_geometric.nn import GroupAddRev, SAGEConv
from torch_geometric.nn.dense.linear import Linear


@pytest.mark.parametrize("num_groups", [2, 4, 8, 16])
def test_revgnn(num_groups):
    hidden_channels = 32
    x = torch.randn(4, hidden_channels)
    edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    seed_gnn = SAGEConv(
        hidden_channels // num_groups,
        hidden_channels // num_groups,
        aggr="mean",
    )
    gnn = GroupAddRev(seed_gnn, num_groups=num_groups, disable=False)
    assert str(gnn) == f"GroupAddRev(num_groups={num_groups}, \
            disable=False"

    lin = Linear(32, 32)

    # Test forward and inverse
    h = lin(x)
    h_o = h.clone().detach()
    out = gnn(h, edge_index)
    assert h.storage().size() == 0
    h_rev = gnn.inverse(out, edge_index)
    assert torch.allclose(h_o, h_rev, atol=0.001)

    # Test backward
    h = lin(x)
    h_o = h.clone().detach()
    out = gnn(h, edge_index)
    target = out.mean()
    target.backward()

    # Test multi-backward
    gnn = GroupAddRev(seed_gnn, num_groups=num_groups, disable=False,
                      num_bwd_passes=4)
    h = lin(x)
    h_o = h.clone().detach()
    out = gnn(h, edge_index)
    target = out.mean()
    target.backward(retain_graph=True)
    target.backward(retain_graph=True)
    torch.autograd.grad(outputs=target, inputs=[h] + list(gnn.parameters()),
                        retain_graph=True)
    torch.autograd.grad(outputs=target, inputs=[h] + list(gnn.parameters()))

    # Test disable
    gnn.disable = True
    h = lin(x)
    h_o = h.clone().detach()
    out = gnn(h, edge_index)
    target = out.mean()
    target.backward()
    assert h.storage().size() == 4 * 32
