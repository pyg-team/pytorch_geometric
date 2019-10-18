import torch
from torch_geometric.nn import BatchNorm


def test_batch_norm():
    norm = BatchNorm(16)
    assert norm.__repr__() == (
        'BatchNorm(16, eps=1e-05, momentum=0.1, affine=True, '
        'track_running_stats=True)')
    out = norm(torch.randn(100, 16))
    assert out.size() == (100, 16)

    norm = BatchNorm(16, affine=False, track_running_stats=False)
    out = norm(torch.randn(100, 16))
    assert out.size() == (100, 16)
