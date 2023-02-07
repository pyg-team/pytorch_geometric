import torch

from torch_geometric.data import Data
from torch_geometric.transforms import RemoveTrainingClasses


def test_remove_training_classes():
    x = torch.randn(6, 8)
    y = torch.tensor([1, 0, 0, 2, 1, 3])
    classes = [0, 1]

    data = Data(x=x, y=y)
    data.train_mask = torch.tensor([0, 0, 1, 1, 1, 1]).to(torch.bool)

    model = RemoveTrainingClasses(classes)
    assert str(model) == f'RemoveTrainingClasses({classes})'

    data = model(data)
    assert data.y[data.train_mask].unique().tolist() == [2, 3]
