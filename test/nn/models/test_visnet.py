import pytest
import torch

from torch_geometric.nn import ViSNet
from torch_geometric.testing import withPackage


@withPackage('torch_cluster')
@pytest.mark.parametrize('kwargs', [
    dict(lmax=2, derivative=True, vecnorm_type=None, vertex=False),
    dict(lmax=1, derivative=False, vecnorm_type='max_min', vertex=True),
])
def test_visnet(kwargs):
    z = torch.randint(1, 10, (20, ))
    pos = torch.randn(20, 3)
    batch = torch.zeros(20, dtype=torch.long)

    model = ViSNet(**kwargs)

    model.reset_parameters()

    energy, forces = model(z, pos, batch)

    assert energy.size() == (1, 1)

    if kwargs['derivative']:
        assert forces.size() == (20, 3)
    else:
        assert forces is None
