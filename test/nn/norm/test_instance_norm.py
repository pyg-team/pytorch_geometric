import copy

import pytest
import torch

from torch_geometric.nn import InstanceNorm
from torch_geometric.testing import is_full_test, withDevice


@withDevice
@pytest.mark.parametrize('affine', [False, True])
def test_instance_norm_running_stats_grad(affine, device):
    norm = InstanceNorm(4, affine=affine, track_running_stats=True,
                        device=device)
    reference = torch.nn.InstanceNorm1d(4, affine=affine,
                                        track_running_stats=True,
                                        device=device)
    batch = torch.arange(3, device=device).repeat_interleave(8)

    def reference_forward(x):
        out = reference(x.view(3, 8, 4).transpose(1, 2))
        return out.transpose(1, 2).reshape(24, 4)

    for _ in range(2):
        x = torch.randn(24, 4, device=device, requires_grad=True)
        ref_x = x.detach().clone().requires_grad_()
        out, ref_out = norm(x, batch), reference_forward(ref_x)
        assert torch.allclose(out, ref_out, atol=1e-6)
        out.square().mean().backward()
        ref_out.square().mean().backward()
        assert torch.allclose(x.grad, ref_x.grad, atol=1e-6)
        assert torch.allclose(norm.running_mean, reference.running_mean)
        assert torch.allclose(norm.running_var, reference.running_var)
        assert not norm.running_mean.requires_grad
        assert not norm.running_var.requires_grad

    copy.deepcopy(norm)  # Also needed by torch.optim.swa_utils.AveragedModel.
    norm.eval()
    reference.eval()
    x = torch.randn(24, 4, device=device, requires_grad=True)
    ref_x = x.detach().clone().requires_grad_()
    out, ref_out = norm(x, batch), reference_forward(ref_x)
    assert torch.allclose(out, ref_out, atol=1e-6)
    out.square().mean().backward()
    ref_out.square().mean().backward()
    assert torch.allclose(x.grad, ref_x.grad, atol=1e-6)


@withDevice
@pytest.mark.parametrize('conf', [True, False])
def test_instance_norm(conf, device):
    batch = torch.zeros(100, dtype=torch.long, device=device)

    x1 = torch.randn(100, 16, device=device)
    x2 = torch.randn(100, 16, device=device)

    norm1 = InstanceNorm(16, affine=conf, track_running_stats=conf,
                         device=device)
    norm2 = InstanceNorm(16, affine=conf, track_running_stats=conf,
                         device=device)
    assert str(norm1) == 'InstanceNorm(16)'

    if is_full_test():
        torch.jit.script(norm1)

    out1 = norm1(x1)
    out2 = norm2(x1, batch)
    assert out1.size() == (100, 16)
    assert torch.allclose(out1, out2, atol=1e-7)
    if conf:
        assert torch.allclose(norm1.running_mean, norm2.running_mean)
        assert torch.allclose(norm1.running_var, norm2.running_var)

    out1 = norm1(x2)
    out2 = norm2(x2, batch)
    assert torch.allclose(out1, out2, atol=1e-7)
    if conf:
        assert torch.allclose(norm1.running_mean, norm2.running_mean)
        assert torch.allclose(norm1.running_var, norm2.running_var)

    norm1.eval()
    norm2.eval()

    out1 = norm1(x1)
    out2 = norm2(x1, batch)
    assert torch.allclose(out1, out2, atol=1e-7)

    out1 = norm1(x2)
    out2 = norm2(x2, batch)
    assert torch.allclose(out1, out2, atol=1e-7)

    out1 = norm2(x1)
    out2 = norm2(x2)
    out3 = norm2(torch.cat([x1, x2], dim=0), torch.cat([batch, batch + 1]))
    assert torch.allclose(out1, out3[:100], atol=1e-7)
    assert torch.allclose(out2, out3[100:], atol=1e-7)
