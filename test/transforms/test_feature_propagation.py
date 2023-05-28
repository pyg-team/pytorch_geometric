import torch

from torch_geometric.data import Data
from torch_geometric.transforms import FeaturePropagation, ToSparseTensor


def test_feature_propagation():
    x = torch.randn(6, 4)
    x[0, 1] = float('nan')
    x[2, 3] = float('nan')
    missing_mask = torch.isnan(x)
    edge_index = torch.tensor([[0, 1, 0, 4, 1, 4, 2, 3, 3, 5],
                               [1, 0, 4, 0, 4, 1, 3, 2, 5, 3]])

    transform = FeaturePropagation(missing_mask)
    assert str(transform) == ('FeaturePropagation(missing_features=8.3%, '
                              'num_iterations=40)')

    data1 = Data(x=x, edge_index=edge_index)
    assert torch.isnan(data1.x).sum() == 2
    data1 = FeaturePropagation(missing_mask)(data1)
    assert torch.isnan(data1.x).sum() == 0
    assert data1.x.size() == x.size()

    data2 = Data(x=x, edge_index=edge_index)
    assert torch.isnan(data2.x).sum() == 2
    data2 = ToSparseTensor()(data2)
    data2 = transform(data2)
    assert torch.isnan(data2.x).sum() == 0
    assert torch.allclose(data1.x, data2.x)
