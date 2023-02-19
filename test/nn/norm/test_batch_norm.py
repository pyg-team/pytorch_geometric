import pytest
import torch

from torch_geometric.nn import BatchNorm
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
