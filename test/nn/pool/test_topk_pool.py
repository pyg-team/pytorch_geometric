import torch

from torch_geometric.nn.pool import TopKPooling
from torch_geometric.nn.pool.topk_pool import filter_adj
from torch_geometric.testing import is_full_test


def test_filter_adj():
    edge_index = torch.tensor([[0, 0, 1, 1, 2, 2, 3, 3],
                               [1, 3, 0, 2, 1, 3, 0, 2]])
    edge_attr = torch.Tensor([1, 2, 3, 4, 5, 6, 7, 8])
    perm = torch.tensor([2, 3])

    out = filter_adj(edge_index, edge_attr, perm)
    assert out[0].tolist() == [[0, 1], [1, 0]]
    assert out[1].tolist() == [6, 8]

    if is_full_test():
        jit = torch.jit.script(filter_adj)

        out = jit(edge_index, edge_attr, perm)
        assert out[0].tolist() == [[0, 1], [1, 0]]
        assert out[1].tolist() == [6, 8]


def test_topk_pooling():
    in_channels = 16
    edge_index = torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
                               [1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2]])
    num_nodes = edge_index.max().item() + 1
    x = torch.randn((num_nodes, in_channels))

    pool1 = TopKPooling(in_channels, ratio=0.5)
    assert str(pool1) == 'TopKPooling(16, ratio=0.5, multiplier=1.0)'
    out1 = pool1(x, edge_index)
    assert out1[0].size() == (num_nodes // 2, in_channels)
    assert out1[1].size() == (2, 2)

    pool2 = TopKPooling(in_channels, ratio=None, min_score=0.1)
    assert str(pool2) == 'TopKPooling(16, min_score=0.1, multiplier=1.0)'
    out2 = pool2(x, edge_index)
    assert out2[0].size(0) <= x.size(0) and out2[0].size(1) == (16)
    assert out2[1].size(0) == 2 and out2[1].size(1) <= edge_index.size(1)

    pool3 = TopKPooling(in_channels, ratio=2)
    assert str(pool3) == 'TopKPooling(16, ratio=2, multiplier=1.0)'
    out3 = pool3(x, edge_index)
    assert out3[0].size() == (2, in_channels)
    assert out3[1].size() == (2, 2)

    if is_full_test():
        jit1 = torch.jit.script(pool1)
        assert torch.allclose(jit1(x, edge_index)[0], out1[0])

        jit2 = torch.jit.script(pool2)
        assert torch.allclose(jit2(x, edge_index)[0], out2[0])

        jit3 = torch.jit.script(pool3)
        assert torch.allclose(jit3(x, edge_index)[0], out3[0])
