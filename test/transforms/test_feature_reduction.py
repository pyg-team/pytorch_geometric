import torch
from torch_geometric.transforms import FeatureReduction
from torch_geometric.data import Data


def test_feature_reduction():
    assert FeatureReduction(10).__repr__() == 'FeatureReduction(10)'
    x = torch.ones(20, 20)
    data = Data(x=x)
    data = FeatureReduction(10)(data)
    assert data.x.shape == torch.Size([20, 10])
