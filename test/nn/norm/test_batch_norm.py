import pytest
import torch

from torch_geometric.nn import BatchNorm, HeteroBatchNorm
from torch_geometric.testing import is_full_test, withCUDA


@withCUDA
@pytest.mark.parametrize('conf', [True, False])
def test_batch_norm(device, conf):
    x = torch.randn(100, 16, device=device)

    norm = BatchNorm(16, affine=conf, track_running_stats=conf).to(device)
    norm.reset_running_stats()
    norm.reset_parameters()
    assert str(norm) == 'BatchNorm(16)'

    if is_full_test():
        torch.jit.script(norm)

    out = norm(x)
    assert out.size() == (100, 16)


def test_batch_norm_single_element():
    x = torch.randn(1, 16)

    norm = BatchNorm(16)
    with pytest.raises(ValueError, match="Expected more than 1 value"):
        norm(x)

    with pytest.raises(ValueError, match="requires 'track_running_stats'"):
        norm = BatchNorm(16, track_running_stats=False,
                         allow_single_element=True)

    norm = BatchNorm(16, track_running_stats=True, allow_single_element=True)
    out = norm(x)
    assert torch.allclose(out, x)


@withCUDA
@pytest.mark.parametrize('conf', [True, False])
def test_hetero_batch_norm(device, conf):
    x = torch.randn((100, 16), device=device)

    # Test single type:
    norm = BatchNorm(16, affine=conf, track_running_stats=conf).to(device)
    expected = norm(x)

    type_vec = torch.zeros(100, dtype=torch.long, device=device)
    norm = HeteroBatchNorm(16, num_types=1, affine=conf,
                           track_running_stats=conf).to(device)
    norm.reset_running_stats()
    norm.reset_parameters()
    assert str(norm) == 'HeteroBatchNorm(16, num_types=1)'

    out = norm(x, type_vec)
    assert out.size() == (100, 16)
    assert torch.allclose(out, expected, atol=1e-3)

    # Test multiple types:
    type_vec = torch.randint(5, (100, ), device=device)
    norm = HeteroBatchNorm(16, num_types=5, affine=conf,
                           track_running_stats=conf).to(device)
    out = norm(x, type_vec)
    assert out.size() == (100, 16)

    for i in range(5):  # Check that mean=0 and std=1 across all types:
        mean = out[type_vec == i].mean()
        std = out[type_vec == i].std(unbiased=False)
        assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-7)
        assert torch.allclose(std, torch.ones_like(std), atol=1e-7)
