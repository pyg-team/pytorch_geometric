import pytest
import torch
from torch_scatter import scatter_mean, scatter_std

from torch_geometric.nn import BatchNorm, HeteroBatchNorm
from torch_geometric.testing import is_full_test


@pytest.mark.parametrize('conf', [True, False])
def test_batch_norm(conf):
    x = torch.randn(100, 16)

    norm = BatchNorm(16, affine=conf, track_running_stats=conf)
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


@pytest.mark.parametrize('conf', [True, False])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_hetero_batch_norm(conf, device):
    x = torch.randn((100, 16), device=device)

    norm = torch.nn.BatchNorm1d(16, device=device)
    real_out = norm(x)

    if not conf:
        # test homogeneous case
        num_types = 1
        types = torch.randint(num_types, (100, ), device=device)
        hetero_norm = HeteroBatchNorm(16, num_types=num_types,
                                      elementwise_affine=conf,
                                      track_running_stats=conf, device=device)
        hetero_out = hetero_norm(x, types)

        assert hetero_out.size() == (100, 16)
        assert torch.allclose(hetero_out, real_out, atol=1e-5)

        # test hetero case
        num_types = 5
        types = torch.randint(num_types, (100, ), device=device)
        hetero_norm = HeteroBatchNorm(16, num_types=num_types,
                                      elementwise_affine=conf,
                                      track_running_stats=conf, device=device)
        hetero_out = hetero_norm(x, types)

        # check if means are zero and vars are 1 across all types
        means = scatter_mean(hetero_out, types, dim=0)
        stds = scatter_std(hetero_out, types, dim=0, unbiased=False)

        assert torch.allclose(means, torch.zeros_like(means), atol=1e-4)
        assert torch.allclose(stds, torch.ones_like(stds), atol=1e-4)

    else:
        # test hetero case
        num_types = 5
        types = torch.randint(num_types, (100, ), device=device)
        hetero_norm = HeteroBatchNorm(16, num_types=num_types,
                                      elementwise_affine=conf,
                                      track_running_stats=conf, device=device)

        assert hetero_norm.running_mean.size() == (num_types, 16)
        assert hetero_norm.running_var.size() == (num_types, 16)
