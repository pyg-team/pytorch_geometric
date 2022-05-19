import torch

from torch_geometric.utils import unbatch


def test_unbatch():
    src = torch.arange(10)
    batch = torch.tensor([0, 0, 0, 1, 1, 2, 2, 3, 4, 4])

    out = unbatch(src, batch)
    assert len(out) == 5
    for i in range(len(out)):
        assert torch.equal(out[i], src[batch == i])
