import torch

from torch_geometric.data import Data
from torch_geometric.transforms import RemoveTrainingClasses


def test_remove_training_classes():
    y = torch.tensor([1, 0, 0, 2, 1, 3])

    data = Data(y=y)
    data.train_mask = torch.tensor([0, 0, 1, 1, 1, 1]).to(torch.bool)

    classes = [0, 1]
    transform = RemoveTrainingClasses(classes)
    assert str(transform) == f'RemoveTrainingClasses({classes})'

    data = transform(data)
    assert data.y[data.train_mask].unique().tolist() == [2, 3]
