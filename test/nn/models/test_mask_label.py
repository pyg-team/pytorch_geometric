import torch
from torch.testing import assert_allclose

from torch_geometric.nn import MaskLabel


def test_mask_label():
    model = MaskLabel(2, 10)

    x = torch.rand(4, 10)
    y = torch.Tensor([1, 0, 1, 0]).type(torch.int64)
    mask = torch.Tensor([0, 0, 1, 1]).type(torch.bool)

    out = model(x, y, mask)
    assert_allclose(out[~mask], x[~mask])

    model = MaskLabel(2, 10, method='concat')
    out = model(x, y, mask)
    assert out.shape == (4, 20)

    input = torch.Tensor([True, True, True]).type(torch.bool)
    output = torch.Tensor([False, True, True]).type(torch.bool)

    assert_allclose(MaskLabel.ratio_mask(input, 0.4, False), output)
    assert int(MaskLabel.ratio_mask(input, 0.4, True).sum()) == 2

    input = torch.Tensor([True, True, True]).type(torch.bool)
    output = torch.Tensor([False, False, True]).type(torch.bool)

    assert_allclose(MaskLabel.ratio_mask(input, 0.7, False), output)
    assert int(MaskLabel.ratio_mask(input, 0.7, True).sum()) == 1

    input = torch.Tensor([True, True, True, True]).type(torch.bool)
    output = torch.Tensor([False, False, True, True]).type(torch.bool)

    assert_allclose(MaskLabel.ratio_mask(input, 0.5, False), output)
    assert int(MaskLabel.ratio_mask(input, 0.5, True).sum()) == 2
