import torch

from torch_geometric.data import Data
from torch_geometric.transforms import SVDFeatureReduction


def test_svd_feature_reduction():
    assert str(SVDFeatureReduction(10)) == 'SVDFeatureReduction(10)'

    x = torch.randn(4, 16)
    U, S, _ = torch.linalg.svd(x)
    data = Data(x=x)
    data = SVDFeatureReduction(10)(data)
    assert torch.allclose(data.x, torch.mm(U[:, :10], torch.diag(S[:10])))

    x = torch.randn(4, 8)
    data.x = x
    data = SVDFeatureReduction(10)(Data(x=x))
    assert torch.allclose(data.x, x)
