import torch

from torch_geometric.nn import MeanSubtractionNorm


def test_mean_subtraction_norm():
    x = torch.randn(6, 16)
    index = torch.tensor([0, 0, 1, 1, 1, 2])

    norm = MeanSubtractionNorm()
    out = norm(x)
    assert out.shape == (6, 16)
    assert torch.allclose(out.mean(), torch.tensor(0.), atol=1e-6)
    out = norm(x, index)
    assert out.shape == (6, 16)
    assert torch.allclose(out[0:2].mean(), torch.tensor(0.), atol=1e-6)
    assert torch.allclose(out[0:2].mean(), torch.tensor(0.), atol=1e-6)
