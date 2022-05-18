import pytest
import torch

from torch_geometric.nn import GroupAddRev, SAGEConv
from torch_geometric.nn.dense.linear import Linear

num_groups = [2, 4, 8, 16]


@pytest.mark.parametrize('num_groups', num_groups)
def test_revgnn(num_groups):
    hidden_channels = 32
    x = torch.randn(4, 32)
    edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    seed_gnn = SAGEConv(
        hidden_channels // num_groups,
        hidden_channels // num_groups,
        aggr='mean',
    )
    gnn = GroupAddRev(seed_gnn, num_groups=num_groups, disable=False)
    lin = Linear(32, 32)

    h = lin(x)
    h_o = h.clone().detach()
    out = gnn(h, edge_index)
    assert h.storage().size() == 0
    h_rev = gnn._inverse(out, edge_index)
    assert torch.allclose(h_o, h_rev, atol=0.001)

    gnn.disable = True
    h = lin(x)
    h_o = h.clone().detach()
    out = gnn(h, edge_index)
    assert h.storage().size() == 4 * 32
