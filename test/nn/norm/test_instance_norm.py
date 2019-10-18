import torch
from torch_geometric.nn import InstanceNorm


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
