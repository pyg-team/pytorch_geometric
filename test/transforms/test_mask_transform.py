import pytest
import torch

from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import IndexToMask, MaskToIndex


def check_attr(data, attr, expected_numel, expected_dtype):
    assert hasattr(data, attr)
    assert data[attr].numel() == expected_numel
    assert data[attr].dtype == expected_dtype


def test_index_to_mask():
    assert IndexToMask().__repr__(
    ) == 'IndexToMask(attrs=None, size=None, replace=False)'

    train_idx = torch.arange(0, 10)
    val_idx = torch.arange(10, 15)
    test_idx = torch.arange(15, 20)
    data = Data(train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
                num_nodes=20)

    out = IndexToMask()(data.clone())
    assert len(out) == len(data) + 3

    for attr in ["train_mask", "val_mask", "test_mask"]:
        check_attr(out, attr, 20, torch.bool)

    out = IndexToMask(sizes=30)(data.clone())
    assert len(out) == len(data) + 3

    for attr in ["train_mask", "val_mask", "test_mask"]:
        check_attr(out, attr, 30, torch.bool)

    out = IndexToMask(replace=True)(data.clone())
    assert len(out) == len(data)

    for attr in ["train_idx", "val_idx", "test_idx"]:
        check_attr(out, attr, 20, torch.bool)

    out = IndexToMask(attrs="train_idx")(data.clone())
    assert len(out) == len(data) + 1
    check_attr(out, "train_mask", 20, torch.bool)

    with pytest.raises(ValueError):
        IndexToMask(sizes=(1, 2))(data)


def test_mask_to_index():
    assert MaskToIndex().__repr__() == 'MaskToIndex(attrs=None, replace=False)'

    train_mask = torch.tensor([1, 0, 1, 1, 0, 0, 0]).bool()
    val_mask = torch.tensor([0, 1, 0, 0, 1, 0, 0]).bool()
    test_mask = torch.tensor([0, 0, 0, 0, 0, 1, 1]).bool()
    data = Data(train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

    out = MaskToIndex()(data.clone())
    assert len(out) == len(data) + 3
    check_attr(out, "train_idx", 3, torch.long)
    check_attr(out, "val_idx", 2, torch.long)
    check_attr(out, "test_idx", 2, torch.long)

    out = MaskToIndex(attrs="train_mask")(data.clone())
    assert len(out) == len(data) + 1
    check_attr(out, "train_idx", 3, torch.long)

    out = MaskToIndex(replace=True)(data.clone())
    assert len(out) == len(data)
    check_attr(out, "train_mask", 3, torch.long)


def test_hetero_index_to_mask():
    data = HeteroData()
    data['u'].train_idx = torch.arange(0, 10)
    data['u'].val_idx = torch.arange(10, 15)
    data['u'].test_idx = torch.arange(15, 25)
    data['u'].num_nodes = 25

    data['v'].train_idx = torch.arange(0, 10)
    data['v'].val_idx = torch.arange(10, 15)
    data['v'].test_idx = torch.arange(15, 20)
    data['v'].num_nodes = 20

    out = IndexToMask()(data.clone())
    assert len(out) == len(data) + 3

    for attr in ["train_mask", "val_mask", "test_mask"]:
        check_attr(out['u'], attr, 25, torch.bool)
        check_attr(out['v'], attr, 20, torch.bool)


def test_hetero_index_to_mask():
    data = HeteroData()

    data['u'].train_mask = torch.tensor([1, 0, 1, 1, 0, 0, 0]).bool()
    data['u'].val_mask = torch.tensor([0, 1, 0, 0, 1, 0, 0]).bool()
    data['u'].test_mask = torch.tensor([0, 0, 0, 0, 0, 1, 1]).bool()

    data['v'].train_mask = torch.tensor([1, 0, 1, 1, 0, 0, 0]).bool()
    data['v'].val_mask = torch.tensor([0, 1, 0, 0, 1, 0, 0]).bool()
    data['v'].test_mask = torch.tensor([0, 0, 0, 0, 0, 1, 1]).bool()

    out = MaskToIndex()(data.clone())
    assert len(out) == len(data) + 3

    for type in ['u', 'v']:
        check_attr(out[type], "train_idx", 3, torch.long)
        check_attr(out[type], "val_idx", 2, torch.long)
        check_attr(out[type], "test_idx", 2, torch.long)
