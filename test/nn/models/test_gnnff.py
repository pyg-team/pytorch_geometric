import torch
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.nn import GNNFF
from torch_geometric.testing import is_full_test


def test_gnnff():
    data = Data(
        z=torch.randint(1, 10, (20, )),
        pos=torch.randn(20, 3),
        force=torch.rand(20, 3),
    )

    model = GNNFF(
        hidden_node_channels=5,
        hidden_edge_channels=5,
        num_message_passings=5,
    )

    with torch.no_grad():
        out = model(data.z, data.pos)
        assert out.size() == (20, 3)

        if is_full_test():
            jit = torch.jit.export(model)
            assert torch.allclose(jit(data.z, data.pos), out)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    for i in range(100):
        optimizer.zero_grad()
        out = model(data.z, data.pos)
        loss = F.l1_loss(out, data.force)
        loss.backward()
        optimizer.step()
    assert loss < 2
