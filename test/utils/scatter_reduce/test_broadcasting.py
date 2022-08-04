from itertools import product

import pytest
import torch
from torch_geometric.utils import scatter
from torch_geometric.utils.torch_version import torch_version_minor

from .utils import reductions, devices


@pytest.mark.skipif(torch_version_minor() < 12,
                    reason='requires torch=1.12.0 or higher')
@pytest.mark.parametrize('reduce,device', product(reductions, devices))
def test_broadcasting(reduce, device):
    B, C, H, W = (4, 3, 8, 8)

    src = torch.randn((B, C, H, W), device=device)
    index = torch.randint(0, H, (H, )).to(device, torch.long)
    out = scatter(src, index, dim=2, dim_size=H, reduce=reduce)
    assert out.size() == (B, C, H, W)

    src = torch.randn((B, C, H, W), device=device)
    index = torch.randint(0, H, (B, 1, H, W)).to(device, torch.long)
    out = scatter(src, index, dim=2, dim_size=H, reduce=reduce)
    assert out.size() == (B, C, H, W)

    src = torch.randn((B, C, H, W), device=device)
    index = torch.randint(0, H, (H, )).to(device, torch.long)
    out = scatter(src, index, dim=2, dim_size=H, reduce=reduce)
    assert out.size() == (B, C, H, W)
