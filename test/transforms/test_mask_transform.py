import copy

import torch

from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import IndexToMask, MaskToIndex


def test_index_to_mask():
    assert str(IndexToMask()) == ('IndexToMask(attrs=None, sizes=None, '
                                  'replace=False)')

    train_index = torch.arange(0, 3)
    test_index = torch.arange(3, 5)
    data = Data(train_index=train_index, test_index=test_index, num_nodes=5)

    out = IndexToMask(replace=True)(copy.copy(data))
    assert len(out) == len(data)
    assert out.train_mask.tolist() == [True, True, True, False, False]
    assert out.test_mask.tolist() == [False, False, False, True, True]

    out = IndexToMask(replace=False)(copy.copy(data))
    assert len(out) == len(data) + 2

    out = IndexToMask(sizes=6, replace=True)(copy.copy(data))
    assert out.train_mask.tolist() == [True, True, True, False, False, False]
    assert out.test_mask.tolist() == [False, False, False, True, True, False]

    out = IndexToMask(attrs='train_index')(copy.copy(data))
    assert len(out) == len(data) + 1
    assert 'train_index' in out
    assert 'train_mask' in out
    assert 'test_index' in out
    assert 'test_mask' not in out


def test_mask_to_index():
    assert str(MaskToIndex()) == 'MaskToIndex(attrs=None, replace=False)'

    train_mask = torch.tensor([True, True, True, False, False])
    test_mask = torch.tensor([False, False, False, True, True])
    data = Data(train_mask=train_mask, test_mask=test_mask)

    out = MaskToIndex(replace=True)(copy.copy(data))
    assert len(out) == len(data)
    assert out.train_index.tolist() == [0, 1, 2]
    assert out.test_index.tolist() == [3, 4]

    out = MaskToIndex(replace=False)(copy.copy(data))
    assert len(out) == len(data) + 2

    out = MaskToIndex(attrs='train_mask')(copy.copy(data))
    assert len(out) == len(data) + 1
    assert 'train_mask' in out
    assert 'train_index' in out
    assert 'test_mask' in out
    assert 'test_index' not in out


def test_hetero_index_to_mask():
    data = HeteroData()
    data['u'].train_index = torch.arange(0, 3)
    data['u'].test_index = torch.arange(3, 5)
    data['u'].num_nodes = 5

    data['v'].train_index = torch.arange(0, 3)
    data['v'].test_index = torch.arange(3, 5)
    data['v'].num_nodes = 5

    out = IndexToMask()(copy.copy(data))
    assert len(out) == len(data) + 2
    assert 'train_mask' in out['u']
    assert 'test_mask' in out['u']
    assert 'train_mask' in out['v']
    assert 'test_mask' in out['v']


def test_hetero_mask_to_index():
    data = HeteroData()
    data['u'].train_mask = torch.tensor([True, True, True, False, False])
    data['u'].test_mask = torch.tensor([False, False, False, True, True])

    data['v'].train_mask = torch.tensor([True, True, True, False, False])
    data['v'].test_mask = torch.tensor([False, False, False, True, True])

    out = MaskToIndex()(copy.copy(data))
    assert len(out) == len(data) + 2
    assert 'train_index' in out['u']
    assert 'test_index' in out['u']
    assert 'train_index' in out['v']
    assert 'test_index' in out['v']
