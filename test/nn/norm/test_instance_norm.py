import torch
from torch_geometric.nn import InstanceNorm, BatchNorm


def test_instance_norm():
    batch = torch.repeat_interleave(torch.full((10, ), 10, dtype=torch.long))

    norm = InstanceNorm(16)
    assert norm.__repr__() == (
        'InstanceNorm(16, eps=1e-05, momentum=0.1, affine=False, '
        'track_running_stats=False)')
    out = norm(torch.randn(100, 16), batch)
    assert out.size() == (100, 16)

    norm = InstanceNorm(16, affine=True, track_running_stats=True)
    out = norm(torch.randn(100, 16), batch)
    assert out.size() == (100, 16)

    # Should behave equally to `BatchNorm` for mini-batches of size 1.
    x = torch.randn(100, 16)
    norm1 = InstanceNorm(16, affine=False, track_running_stats=False)
    norm2 = BatchNorm(16, affine=False, track_running_stats=False)
    assert torch.allclose(norm1(x), norm2(x), atol=1e-6)

    norm1 = InstanceNorm(16, affine=False, track_running_stats=True)
    norm2 = BatchNorm(16, affine=False, track_running_stats=True)
    assert torch.allclose(norm1(x), norm2(x), atol=1e-6)
    assert torch.allclose(norm1.running_mean, norm2.running_mean, atol=1e-6)
    assert torch.allclose(norm1.running_var, norm2.running_var, atol=1e-6)
    assert torch.allclose(norm1(x), norm2(x), atol=1e-6)
    assert torch.allclose(norm1.running_mean, norm2.running_mean, atol=1e-6)
    assert torch.allclose(norm1.running_var, norm2.running_var, atol=1e-6)
    norm1.eval()
    norm2.eval()
    assert torch.allclose(norm1(x), norm2(x), atol=1e-6)
