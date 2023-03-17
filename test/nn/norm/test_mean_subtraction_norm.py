import torch

from torch_geometric.nn import MeanSubtractionNorm
from torch_geometric.testing import is_full_test


def test_mean_subtraction_norm():
    x = torch.randn(6, 16)
    batch = torch.tensor([0, 0, 1, 1, 1, 2])

    norm = MeanSubtractionNorm()
    assert str(norm) == 'MeanSubtractionNorm()'

    if is_full_test():
        torch.jit.script(norm)

    out = norm(x)
    assert out.size() == (6, 16)
    assert torch.allclose(out.mean(), torch.tensor(0.), atol=1e-6)

    out = norm(x, batch)
    assert out.size() == (6, 16)
    assert torch.allclose(out[0:2].mean(), torch.tensor(0.), atol=1e-6)
    assert torch.allclose(out[0:2].mean(), torch.tensor(0.), atol=1e-6)
