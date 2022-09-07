import pytest
import torch

from torch_geometric.data import Data
from torch_geometric.transforms import IndexToMask, MaskToIndex


def test_index_to_mask():
    assert IndexToMask().__repr__() == 'IndexToMask(attrs=None, size=None)'

    train_indices = torch.arange(0, 10)
    val_indices = torch.arange(10, 15)
    test_indices = torch.arange(15, 20)
    data = Data(train_indices=train_indices, val_indices=val_indices,
                test_indices=test_indices)

    out = IndexToMask()(data.clone())
    assert len(out) == 6
    assert hasattr(out, "train_mask")
    assert hasattr(out, "val_mask")
    assert hasattr(out, "test_mask")

    out = IndexToMask(sizes=20)(data.clone())
    assert out.train_mask.numel() == 20
    assert out.val_mask.numel() == 20
    assert out.test_mask.numel() == 20

    out = IndexToMask(attrs="train_indices")(data.clone())
    assert len(out) == 4

    with pytest.raises(ValueError):
        IndexToMask(sizes=(1, 2))(data)


def test_mask_to_index():
    assert MaskToIndex().__repr__() == 'MaskToIndex(attrs=None)'

    train_mask = torch.tensor([1, 0, 1, 1, 0, 0, 0]).bool()
    val_mask = torch.tensor([0, 1, 0, 0, 1, 0, 0]).bool()
    test_mask = torch.tensor([0, 0, 0, 0, 0, 1, 1]).bool()
    data = Data(train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

    out = MaskToIndex()(data.clone())
    assert len(out) == 6
    assert hasattr(out, "train_indices")
    assert hasattr(out, "val_indices")
    assert hasattr(out, "test_indices")

    out = MaskToIndex(attrs="train_mask")(data.clone())
    assert len(out) == 4
    assert out.train_indices.numel() == 3
