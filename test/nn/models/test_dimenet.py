import torch
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.nn import DimeNetPlusPlus
from torch_geometric.testing import is_full_test, onlyFullTest


@onlyFullTest
def test_dimenet_plus_plus():
    data = Data(
        z=torch.randint(1, 10, (20, )),
        pos=torch.randn(20, 3),
        y=torch.tensor([1.]),
    )

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

    with torch.no_grad():
        out = model(data.z, data.pos)
        assert out.size() == (1, )

        if is_full_test():
            jit = torch.jit.export(model)
            assert torch.allclose(jit(data.z, data.pos), out)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    for i in range(100):
        optimizer.zero_grad()
        out = model(data.z, data.pos)
        loss = F.l1_loss(out, data.y)
        loss.backward()
        optimizer.step()
    assert loss < 2
