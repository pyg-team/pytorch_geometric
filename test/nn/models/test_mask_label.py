import torch

from torch_geometric.nn import MaskLabel


def test_mask_label():
    model = MaskLabel(2, 10)
    assert str(model) == 'MaskLabel()'

    x = torch.rand(4, 10)
    y = torch.tensor([1, 0, 1, 0])
    mask = torch.tensor([False, False, True, True])

    out = model(x, y, mask)
    assert out.size() == (4, 10)
    assert torch.allclose(out[~mask], x[~mask])

    model = MaskLabel(2, 10, method='concat')
    out = model(x, y, mask)
    assert out.size() == (4, 20)
    assert torch.allclose(out[:, :10], x)


def test_ratio_mask():
    mask = torch.tensor([True, True, True])
    expected = [False, True, True]
    assert MaskLabel.ratio_mask(mask, 0.4, False).tolist() == expected
    assert MaskLabel.ratio_mask(mask, 0.4, True).sum() == 2

    mask = torch.tensor([True, True, True])
    expected = [False, False, True]
    assert MaskLabel.ratio_mask(mask, 0.7, False).tolist() == expected
    assert MaskLabel.ratio_mask(mask, 0.7, True).sum() == 1

    mask = torch.tensor([True, True, True, True])
    expected = [False, False, True, True]
    assert MaskLabel.ratio_mask(mask, 0.5, False).tolist() == expected
    assert int(MaskLabel.ratio_mask(mask, 0.5, True).sum()) == 2
