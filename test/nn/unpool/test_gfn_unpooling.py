import torch

from torch_geometric.nn import GFNUnpooling
from torch_geometric.testing import withPackage


def _setup_gfn_unpooling():
    weight = torch.tensor(
        [[1, 10, 100], [2, 20, 200], [3, 30, 300], [4, 40, 400]], dtype=float)
    bias = torch.tensor([1, 10, 100, 1000], dtype=float)

    out_graph = torch.tensor([(-3, 2), (1, 0), (2, 1), (3, 2)])

    unpool = GFNUnpooling(3, out_graph)

    with torch.no_grad():
        unpool._gfn.weight.copy_(weight)
        unpool._gfn.bias.copy_(bias)

    return unpool


@withPackage('gfn')
def test_gfn_unpooling():
    unpool = _setup_gfn_unpooling()

    x = torch.tensor([[1.0], [10.0], [100.0], [-1.0], [-10.0], [-100.0]])
    pos_y = torch.tensor([
        [-1.0, -1.0],
        [1.0, 1.0],
        [-2.0, -2.0],
        [2.0, 2.0],
    ])
    batch_x = torch.tensor([0, 0, 0, 1, 1, 1])
    batch_y = torch.tensor([0, 0, 1, 1])

    y = unpool(x, pos_y, batch_x, batch_y)

    expected = torch.tensor([[15157.], [30673.], [-15146.], [-29933.]])

    torch.testing.assert_close(y, expected)
