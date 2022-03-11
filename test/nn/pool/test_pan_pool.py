import torch

from torch_geometric.nn import PANConv, PANPooling


def test_pan_pooling():
    edge_index = torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
                               [1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2]])
    num_nodes = edge_index.max().item() + 1
    x = torch.randn((num_nodes, 16))

    conv = PANConv(16, 32, filter_size=2)
    pool = PANPooling(32, ratio=0.5)
    assert pool.__repr__() == 'PANPooling(32, ratio=0.5, multiplier=1.0)'

    x, M = conv(x, edge_index)
    x, edge_index, edge_weight, batch, perm, score = pool(x, M)

    assert x.size() == (2, 32)
    assert edge_index.size() == (2, 4)
    assert edge_weight.size() == (4, )
    assert perm.size() == (2, )
    assert score.size() == (2, )
