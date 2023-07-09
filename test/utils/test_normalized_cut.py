import torch

from torch_geometric.testing import is_full_test
from torch_geometric.utils import normalized_cut


def test_normalized_cut():
    row = torch.tensor([0, 1, 1, 1, 2, 2, 3, 3, 4, 4])
    col = torch.tensor([1, 0, 2, 3, 1, 4, 1, 4, 2, 3])
    edge_attr = torch.tensor(
        [3.0, 3.0, 6.0, 3.0, 6.0, 1.0, 3.0, 2.0, 1.0, 2.0])
    expected = torch.tensor([4.0, 4.0, 5.0, 2.5, 5.0, 1.0, 2.5, 2.0, 1.0, 2.0])

    out = normalized_cut(torch.stack([row, col], dim=0), edge_attr)
    assert torch.allclose(out, expected)

    if is_full_test():
        jit = torch.jit.script(normalized_cut)
        out = jit(torch.stack([row, col], dim=0), edge_attr)
        assert torch.allclose(out, expected)
