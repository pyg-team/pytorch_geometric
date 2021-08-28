import torch
from torch_geometric_test.transforms import GetzsTrainMask
from torch_geometric.data import Data


def test_get_zs_train_mask():
    assert GetzsTrainMask({0, 1, 2}).__repr__() == 'GetzsTrainMask({0, 1, 2})'
    y = torch.Tensor([0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6])
    train_mask = torch.BoolTensor([True, True, True, True,
                                   True, True, True, False,
                                   False, False, False, False, False, False])
    data = Data(y=y, train_mask=train_mask)
    data = GetzsTrainMask({0, 1, 2})(data)
    assert sum(data.train_mask).item() == 4
    assert data.y[data.train_mask].equal(torch.Tensor([3, 4, 5, 6]))
