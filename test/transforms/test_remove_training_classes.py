import torch

from torch_geometric.data import Data
from torch_geometric.transforms import RemoveTrainingClasses


def test_remove_training_classes():
    y = torch.tensor([1, 0, 0, 2, 1, 3])
    train_mask = torch.tensor([False, False, True, True, True, True])

    data = Data(y=y, train_mask=train_mask)

    transform = RemoveTrainingClasses(classes=[0, 1])
    assert str(transform) == 'RemoveTrainingClasses([0, 1])'

    data = transform(data)
    assert len(data) == 2
    assert torch.equal(data.y, y)
    assert data.train_mask.tolist() == [False, False, False, True, False, True]
