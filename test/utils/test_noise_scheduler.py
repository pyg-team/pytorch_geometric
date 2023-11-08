import numpy as np
import pytest

from torch_geometric.utils.noise_scheduler import (
    get_diffusion_beta_schedule,
    get_smld_sigma_schedule,
)


def test_get_smld_sigma_schedule():
    sigma_min = 0.01
    sigma_max = 1.0
    num_scales = 10
    expected_sigmas = np.array([
        1., 0.59948425, 0.35938137, 0.21544347, 0.12915497, 0.07742637,
        0.04641589, 0.02782559, 0.01668101, 0.01
    ])
    generated_sigmas = get_smld_sigma_schedule(sigma_min, sigma_max,
                                               num_scales)
    np.testing.assert_allclose(generated_sigmas, expected_sigmas, rtol=1e-5)


def test_get_diffusion_beta_schedule_linear():
    schedule_type = "linear"
    beta_start = 0.1
    beta_end = 0.2
    num_diffusion_timesteps = 10
    expected_betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps)
    generated_betas = get_diffusion_beta_schedule(schedule_type, beta_start,
                                                  beta_end,
                                                  num_diffusion_timesteps)
    np.testing.assert_allclose(generated_betas, expected_betas, rtol=1e-5)


def test_get_diffusion_beta_schedule_const():
    beta_end = 0.2
    num_diffusion_timesteps = 10
    expected = beta_end * np.ones(num_diffusion_timesteps)
    np.testing.assert_allclose(
        get_diffusion_beta_schedule("const", 0, beta_end,
                                    num_diffusion_timesteps), expected,
        rtol=1e-5)


def test_get_diffusion_beta_schedule_quad():
    betas = get_diffusion_beta_schedule("quad", 0.1, 0.2, 10)
    expected_betas = np.linspace(np.sqrt(0.1), np.sqrt(0.2), 10)**2
    np.testing.assert_allclose(betas, expected_betas)


def test_get_diffusion_beta_schedule_sigmoid():
    betas = get_diffusion_beta_schedule("sigmoid", 0.1, 0.2, 10)
    expected_betas = np.linspace(-6, 6, 10)
    expected_betas = 1 / (1 + np.exp(-expected_betas))
    expected_betas = expected_betas * (0.2 - 0.1) + 0.1
    np.testing.assert_allclose(betas, expected_betas)


def test_diffusion_beta_schedule_invalid_type():
    with pytest.raises(NotImplementedError):
        get_diffusion_beta_schedule("invalid_type", 0.1, 0.2, 10)
