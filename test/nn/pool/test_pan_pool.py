import torch

import torch_geometric.typing
from torch_geometric.nn import PANConv, PANPooling
from torch_geometric.testing import is_full_test, withPackage


@withPackage('torch_sparse')
def test_pan_pooling():
    edge_index = torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
                               [1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2]])
    num_nodes = edge_index.max().item() + 1
    x = torch.randn((num_nodes, 16))

    conv = PANConv(16, 32, filter_size=2)
    pool = PANPooling(32, ratio=0.5)
    assert str(pool) == 'PANPooling(32, ratio=0.5, multiplier=1.0)'

    x, M = conv(x, edge_index)
    h, edge_index, edge_weight, batch, perm, score = pool(x, M)

    assert h.size() == (2, 32)
    assert edge_index.size() == (2, 4)
    assert edge_weight.size() == (4, )
    assert perm.size() == (2, )
    assert score.size() == (2, )

    if torch_geometric.typing.WITH_PT112 and is_full_test():
        jit = torch.jit.script(pool)
        out = jit(x, M)
        assert torch.allclose(h, out[0])
        assert torch.equal(edge_index, out[1])
        assert torch.allclose(edge_weight, out[2])
        assert torch.equal(batch, out[3])
        assert torch.equal(perm, out[4])
        assert torch.allclose(score, out[5])
