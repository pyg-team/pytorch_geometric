import pytest
import torch

from torch_geometric.testing import withCUDA
from torch_geometric.utils import segment


@withCUDA
@pytest.mark.parametrize('reduce', ['sum', 'mean', 'min', 'max'])
def test_segment(device, reduce):
    src = torch.randn(20, 16, device=device)
    ptr = torch.tensor([0, 5, 10, 15, 20], device=device)

    out = segment(src, ptr, reduce=reduce)

    expected = getattr(torch, reduce)(src.view(4, 5, -1), dim=1)
    expected = expected[0] if isinstance(expected, tuple) else expected

    assert torch.allclose(out, expected)
