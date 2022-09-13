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

    assert str(FeaturePropagation()) == ('FeaturePropagation('
                                         'num_iterations=40)')

    data = Data(x=x, edge_index=edge_index)
    assert torch.any(torch.isnan(data.x))

    data = FeaturePropagation(missing_mask)(data)
    assert torch.all(~torch.isnan(data.x))
    assert data.x.size() == x.size()

    data = Data(x=x, edge_index=edge_index)
    data = ToSparseTensor()(data)
    data = FeaturePropagation(missing_mask)(data)
    assert torch.all(~torch.isnan(data.x))
    assert data.x.size() == x.size()
