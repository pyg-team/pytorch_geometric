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
    assert hasattr(out, "train_mask")
    assert hasattr(out, "val_mask")
    assert hasattr(out, "test_mask")

    out = IndexToMask(sizes=30)(data.clone())
    assert out.train_mask.numel() == 30
    assert out.val_mask.numel() == 30
    assert out.test_mask.numel() == 30

    out = IndexToMask(replace=True)(data.clone())
    assert len(out) == len(data)
    assert out.train_idx.dtype == torch.bool
    assert out.val_idx.dtype == torch.bool
    assert out.test_idx.dtype == torch.bool

    out = IndexToMask(attrs="train_idx")(data.clone())
    assert len(out) == len(data) + 1

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
    assert hasattr(out, "train_idx")
    assert hasattr(out, "val_idx")
    assert hasattr(out, "test_idx")

    out = MaskToIndex(attrs="train_mask")(data.clone())
    assert len(out) == len(data) + 1
    assert out.train_idx.numel() == 3

    out = MaskToIndex(replace=True)(data.clone())
    assert len(out) == len(data)
    assert out.train_mask.dtype == torch.long
