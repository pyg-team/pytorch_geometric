import pytest
import torch

from torch_geometric.nn import ViSNet
from torch_geometric.testing import withPackage

@withPackage('torch_cluster')
@pytest.mark.parametrize('Version', [1, 2])
def test_visnet(Version):
    z = torch.randint(1, 10, (20, ))
    pos = torch.randn(20, 3)
    batch = torch.zeros(20, dtype=torch.long)

    if Version == 1:
        kwargs = dict(lmax=2, derivative=True, vecnorm_type='none', vertex=False)
    else:
        kwargs = dict(lmax=1, derivative=False, vecnorm_type='max_min', vertex=True)

    model = ViSNet(
        trainable_vecnorm=False,
        num_heads=8,
        num_layers=6,
        hidden_channels=128,
        num_rbf=32,
        trainable_rbf=False,
        max_z=100,
        cutoff=5.0,
        max_num_neighbors=32,
        atomref=None,
        reduce_op="add",
        mean=None,
        std=None,
        **kwargs,
    )

    model.reset_parameters()

    energy, forces = model(z, pos, batch)

    assert energy.size() == (1, 1)

    if Version == 1:
        assert forces.size() == (20, 3)
    else:
        assert forces is None
