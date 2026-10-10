import pytest
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

    jit = torch.jit.script(index_to_mask)
    assert torch.equal(jit(index), index_to_mask(index))
    assert torch.equal(jit(index, 7), mask)


@pytest.mark.parametrize('dtype', [torch.int32, torch.int64])
@pytest.mark.parametrize('size', [None, 0, 4])
@pytest.mark.parametrize('script', [False, True])
def test_index_to_mask_empty(dtype, size, script):
    index = torch.empty(0, dtype=dtype)
    func = torch.jit.script(index_to_mask) if script else index_to_mask

    mask = func(index, size)
    assert mask.dtype == torch.bool
    assert mask.device == index.device
    assert mask.shape == (0 if size is None else size, )
    assert not mask.any()


def test_mask_to_index():
    mask = torch.tensor([False, True, False, True, False, True])

    index = mask_to_index(mask)
    assert index.tolist() == [1, 3, 5]
