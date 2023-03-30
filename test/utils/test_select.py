import torch

from torch_geometric.utils import narrow, select


def test_select():
    src = torch.randn(5, 3)
    index = torch.tensor([0, 2, 4])
    mask = torch.tensor([True, False, True, False, True])

    out = select(src, index, dim=0)
    assert torch.equal(out, src[index])
    assert torch.equal(out, select(src, mask, dim=0))
    assert torch.equal(out, torch.tensor(select(src.tolist(), index, dim=0)))
    assert torch.equal(out, torch.tensor(select(src.tolist(), mask, dim=0)))


def test_narrow():
    src = torch.randn(5, 3)

    out = narrow(src, dim=0, start=2, length=2)
    assert torch.equal(out, src[2:4])
    assert torch.equal(out, torch.tensor(narrow(src.tolist(), 0, 2, 2)))
