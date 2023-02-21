import torch
import torch.nn.functional as F

from torch_geometric.nn import DimeNetPlusPlus
from torch_geometric.testing import onlyFullTest


@onlyFullTest
def test_dimenet_plus_plus():
    z = torch.randint(1, 10, (20, ))
    pos = torch.randn(20, 3)

    model = DimeNetPlusPlus(
        hidden_channels=5,
        out_channels=1,
        num_blocks=5,
        out_emb_channels=3,
        int_emb_size=5,
        basis_emb_size=5,
        num_spherical=5,
        num_radial=5,
        num_before_skip=2,
        num_after_skip=2,
    )
    model.reset_parameters()

    with torch.no_grad():
        out = model(z, pos)
        assert out.size() == (1, )

        jit = torch.jit.export(model)
        assert torch.allclose(jit(z, pos), out)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    min_loss = float('inf')
    for i in range(100):
        optimizer.zero_grad()
        out = model(z, pos)
        loss = F.l1_loss(out, torch.tensor([1.]))
        loss.backward()
        optimizer.step()
        min_loss = min(float(loss), min_loss)
    assert min_loss < 2
