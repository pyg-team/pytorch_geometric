import torch

from torch_geometric.data import Data
from torch_geometric.datasets import MedShapeNet
from torch_geometric.testing import withPackage


@withPackage('MedShapeNet')
def test_medshapenet():
    dataset = MedShapeNet(root="./data/MedShapeNet", size=1)

    assert str(dataset) == f'MedShapeNet({len(dataset)})'

    assert isinstance(dataset[0], Data)
    assert dataset.num_classes == 8

    assert isinstance(dataset[0].pos, torch.Tensor)
    assert len(dataset[0].pos) > 0

    assert isinstance(dataset[0].face, torch.Tensor)
    assert len(dataset[0].face) == 3

    assert isinstance(dataset[0].y, torch.Tensor)
    assert len(dataset[0].y) == 1
