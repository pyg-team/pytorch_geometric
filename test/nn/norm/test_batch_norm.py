import pytest
import torch

from torch_geometric.nn import BatchNorm
from torch_geometric.testing import is_full_test


@pytest.mark.parametrize('conf', [True, False])
def test_batch_norm(conf):
    x = torch.randn(100, 16)

    norm = BatchNorm(16, affine=conf, track_running_stats=conf)
    assert norm.__repr__() == 'BatchNorm(16)'

    if is_full_test():
        torch.jit.script(norm)

    out = norm(x)
    assert out.size() == (100, 16)
