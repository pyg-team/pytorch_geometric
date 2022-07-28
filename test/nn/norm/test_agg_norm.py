import pytest
import torch

from torch_geometric.nn import AggSubtraction


def test_agg_norm():
    x = torch.randn(6, 16)
    index = torch.tensor([0, 0, 1, 1, 1, 2])

    norm = AggSubtraction()
    out = norm(x)
    assert out.shape == (6, 16)
    assert out.mean().item() == pytest.approx(0, abs=1e-6)

    out = norm(x, index)
    assert out.shape == (6, 16)
    assert out[0:2].mean().item() == pytest.approx(0, abs=1e-6)
    assert out[0:2].mean().item() == pytest.approx(0, abs=1e-6)
