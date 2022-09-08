import pytest
import torch

from torch_geometric.data import Data
from torch_geometric.transforms import IndexToMask, MaskToIndex


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
        assert hasattr(out, attr)
        assert out[attr].numel() == 20

    out = IndexToMask(sizes=30)(data.clone())
    assert len(out) == len(data) + 3

    for attr in ["train_mask", "val_mask", "test_mask"]:
        assert hasattr(out, attr)
        assert out[attr].numel() == 30

    out = IndexToMask(replace=True)(data.clone())
    assert len(out) == len(data)

    for attr in ["train_idx", "val_idx", "test_idx"]:
        assert out[attr].numel() == 20
        assert out[attr].dtype == torch.bool

    out = IndexToMask(attrs="train_idx")(data.clone())
    assert len(out) == len(data) + 1
    assert hasattr(out, "train_mask")
    assert out.train_mask.numel() == 20

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

    for attr in ["train_idx", "val_idx", "test_idx"]:
        assert hasattr(out, attr)
        assert out[attr].dtype == torch.long

    out = MaskToIndex(attrs="train_mask")(data.clone())
    assert len(out) == len(data) + 1
    assert hasattr(out, "train_idx")
    assert out.train_idx.numel() == 3
    assert out.train_idx.dtype == torch.long

    out = MaskToIndex(replace=True)(data.clone())
    assert len(out) == len(data)
    assert out.train_mask.numel() == 3
    assert out.train_mask.dtype == torch.long
