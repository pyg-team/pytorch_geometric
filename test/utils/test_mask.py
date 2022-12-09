import torch

from torch_geometric.utils import index_to_mask, mask_select, mask_to_index


def test_mask_select():
    src = torch.randn(6, 8)
    mask = torch.tensor([False, True, False, True, False, True])

    out = mask_select(src, 0, mask)
    assert out.size() == (3, 8)
    assert torch.equal(src[torch.tensor([1, 3, 5])], out)

    jit = torch.jit.script(mask_select)
    assert torch.equal(jit(src, 0, mask), out)


def test_index_to_mask():
    index = torch.tensor([1, 3, 5])

    mask = index_to_mask(index)
    assert mask.tolist() == [False, True, False, True, False, True]

    mask = index_to_mask(index, size=7)
    assert mask.tolist() == [False, True, False, True, False, True, False]


def test_mask_to_index():
    mask = torch.tensor([False, True, False, True, False, True])

    index = mask_to_index(mask)
    assert index.tolist() == [1, 3, 5]
