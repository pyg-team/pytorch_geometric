import torch
from torch_geometric.transforms import AddTrainValTestMask
from torch_geometric.data import Data
from copy import deepcopy


def test_add_train_val_test_mask():
    N, C = 2357, 7
    original_data = Data(x=torch.randn(N, 8),
                         y=torch.randint(C, (N,)),
                         num_nodes=N)

    num_train_per_class = 16
    num_val = 320
    num_test = 640

    kw = dict(num_train_per_class=num_train_per_class,
              num_val=num_val, num_test=num_test)

    t = AddTrainValTestMask(split='train-rest', **kw)
    data = t(deepcopy(original_data))
    assert t.__repr__() == 'AddTrainValTestMask(split=train-rest)'
    assert data.train_mask.size(0) == data.val_mask.size(0) \
           == data.test_mask.size(0) == N
    assert data.train_mask.sum() == N - num_val - num_test
    assert data.val_mask.sum() == num_val
    assert data.test_mask.sum() == num_test
    assert (data.train_mask & data.val_mask & data.test_mask).sum() == 0

    t = AddTrainValTestMask(split='test-rest', **kw)
    data = t(deepcopy(original_data))
    assert t.__repr__() == 'AddTrainValTestMask(split=test-rest)'
    assert data.train_mask.size(0) == data.val_mask.size(0) \
           == data.test_mask.size(0) == N
    assert data.train_mask.sum() == num_train_per_class * C
    assert data.val_mask.sum() == num_val
    assert data.test_mask.sum() == N - num_train_per_class * C - num_val
    assert (data.train_mask & data.val_mask & data.test_mask).sum() == 0

    t = AddTrainValTestMask(split='random', **kw)
    data = t(deepcopy(original_data))
    assert t.__repr__() == 'AddTrainValTestMask(split=random)'
    assert data.train_mask.size(0) == data.val_mask.size(0) \
           == data.test_mask.size(0) == N
    assert data.train_mask.sum() == num_train_per_class * C
    assert data.val_mask.sum() == num_val
    assert data.test_mask.sum() == num_test
    assert (data.train_mask & data.val_mask & data.test_mask).sum() == 0

    t = AddTrainValTestMask(split='train-rest', num_val=0.1, num_test=0.1)
    data = t(deepcopy(original_data))
    assert data.train_mask.sum() == N - int(N * 0.1) - int(N * 0.1)
    assert data.val_mask.sum() == int(N * 0.1)
    assert data.test_mask.sum() == int(N * 0.1)
    assert (data.train_mask & data.val_mask & data.test_mask).sum() == 0
