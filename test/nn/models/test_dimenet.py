import pytest
import torch
import torch.nn.functional as F

from torch_geometric.nn import DimeNet, DimeNetPlusPlus
from torch_geometric.nn.models.dimenet import (
    BesselBasisLayer,
    Envelope,
    ResidualLayer,
)
from torch_geometric.testing import is_full_test


def test_dimenet_modules():
    env = Envelope(exponent=5)
    x = torch.randn(10, 3)
    assert env(x).size() == (10, 3)  # Isotonic layer.

    bbl = BesselBasisLayer(5)
    x = torch.randn(10, 3)
    assert bbl(x).size() == (10, 3, 5)  # Non-isotonic layer.

    rl = ResidualLayer(128, torch.nn.functional.relu)
    x = torch.randn(128, 128)
    assert rl(x).size() == (128, 128)  # Isotonic layer.


@pytest.mark.parametrize('Model', [DimeNet, DimeNetPlusPlus])
def test_dimenet(Model):
    z = torch.randint(1, 10, (20, ))
    pos = torch.randn(20, 3)

    if Model == DimeNet:
        kwargs = dict(num_bilinear=3)
    else:
        kwargs = dict(out_emb_channels=3, int_emb_size=5, basis_emb_size=5)

    model = Model(
        hidden_channels=5,
        out_channels=1,
        num_blocks=5,
        num_spherical=5,
        num_radial=5,
        **kwargs,
    )
    model.reset_parameters()

    with torch.no_grad():
        out = model(z, pos)
        assert out.size() == (1, )

        jit = torch.jit.export(model)
        assert torch.allclose(jit(z, pos), out)

    if is_full_test():
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

        min_loss = float('inf')
        for i in range(100):
            optimizer.zero_grad()
            out = model(z, pos)
            loss = F.l1_loss(out, torch.tensor([1.0]))
            loss.backward()
            optimizer.step()
            min_loss = min(float(loss), min_loss)
        assert min_loss < 2
