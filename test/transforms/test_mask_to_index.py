import torch

from torch_geometric.data import Data
from torch_geometric.transforms import MaskToIndex


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
