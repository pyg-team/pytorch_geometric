import torch

from torch_geometric.nn.pool import KMISPooling, cluster_mis
from torch_geometric.testing import is_full_test


def test_cluster_mis():
    index = torch.tensor([[0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 4, 4],
                          [1, 4, 0, 2, 4, 1, 3, 2, 4, 0, 1, 3]]).long()
    rank = torch.arange(5)

    mis1, cluster1 = cluster_mis(index, k=0, rank=rank)
    assert mis1.equal(torch.ones_like(mis1))
    assert cluster1.equal(rank)

    mis2, cluster2 = cluster_mis(index, k=1, rank=rank)
    assert mis2.equal(torch.tensor([1, 0, 1, 0, 0]).bool())
    assert cluster2.equal(torch.tensor([0, 0, 1, 1, 0]).long())

    mis3, cluster3 = cluster_mis(index, k=2, rank=rank)
    assert mis3.equal(torch.tensor([1, 0, 0, 0, 0]).bool())
    assert cluster3.equal(torch.tensor([0, 0, 0, 0, 0]).long())

    if is_full_test():
        jit = torch.jit.script(cluster_mis)
        assert cluster1.equal(jit(index, k=0, rank=rank)[1])
        assert cluster2.equal(jit(index, k=1, rank=rank)[1])
        assert cluster3.equal(jit(index, k=2, rank=rank)[1])


def test_kmis_pooling():
    index = torch.tensor([[0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 4, 4],
                          [1, 4, 0, 2, 4, 1, 3, 2, 4, 0, 1, 3]]).long()

    num_nodes = 5
    in_channels = 16
    x = torch.rand(num_nodes, in_channels)

    pool1 = KMISPooling(in_channels, k=0)
    assert str(pool1) == f'KMISPooling(in_channels={in_channels}, k=0)'
    out1 = pool1(x, index)
    assert out1[0].shape == x.shape
    assert out1[1].equal(index)

    pool2 = KMISPooling(in_channels, k=1)
    assert str(pool2) == f'KMISPooling(in_channels={in_channels}, k=1)'
    out2 = pool2(x, index)
    assert out2[0].size(0) == 2
    assert out2[1].shape == (2, 2)

    pool3 = KMISPooling(in_channels, k=2)
    assert str(pool3) == f'KMISPooling(in_channels={in_channels}, k=2)'
    out3 = pool2(x, index)
    assert out3[0].size(0) == 2
    assert out3[1].shape == (2, 2)

    if is_full_test():
        jit1 = torch.jit.script(pool1)
        assert torch.allclose(jit1(x, index)[0], out1[0])

        jit2 = torch.jit.script(pool2)
        assert torch.allclose(jit2(x, index)[0], out2[0])

        jit3 = torch.jit.script(pool3)
        assert torch.allclose(jit3(x, index)[0], out3[0])
