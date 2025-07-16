import pytest
import torch

from torch_geometric.utils.noise_scheduler import (
    get_diffusion_beta_schedule,
    get_smld_sigma_schedule,
)


def test_get_smld_sigma_schedule():
    expected = torch.tensor([
        1., 0.59948425, 0.35938137, 0.21544347, 0.12915497, 0.07742637,
        0.04641589, 0.02782559, 0.01668101, 0.01
    ])
    out = get_smld_sigma_schedule(
        sigma_min=0.01,
        sigma_max=1.0,
        num_scales=10,
    )
    assert torch.allclose(out, expected)


@pytest.mark.parametrize(
    'schedule_type',
    ['linear', 'quadratic', 'constant', 'sigmoid'],
)
def test_get_diffusion_beta_schedule(schedule_type):
    out = get_diffusion_beta_schedule(
        schedule_type,
        beta_start=0.1,
        beta_end=0.2,
        num_diffusion_timesteps=10,
    )
    assert out.size() == (10, )
