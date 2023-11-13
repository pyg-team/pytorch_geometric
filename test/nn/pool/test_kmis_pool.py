import pytest
import torch

from torch_geometric.nn.aggr import MaxAggregation
from torch_geometric.nn.pool import (
    KMISPooling,
    maximal_independent_set_cluster,
)
from torch_geometric.testing import is_full_test
from torch_geometric.typing import WITH_TORCH_SPARSE


def test_maximal_independent_set_cluster():
    index = torch.tensor([[0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 4, 4],
                          [1, 4, 0, 2, 4, 1, 3, 2, 4, 0, 1, 3]]).long()

    mis1, cluster1 = maximal_independent_set_cluster(index, k=0)
    assert mis1.equal(torch.ones_like(mis1))
    assert cluster1.equal(torch.arange(5))

    mis2, cluster2 = maximal_independent_set_cluster(index, k=1)
    assert mis2.equal(torch.tensor([1, 0, 1, 0, 0]).bool())
    assert cluster2.equal(torch.tensor([0, 0, 1, 1, 0]).long())

    mis3, cluster3 = maximal_independent_set_cluster(index, k=2)
    assert mis3.equal(torch.tensor([1, 0, 0, 0, 0]).bool())
    assert cluster3.equal(torch.tensor([0, 0, 0, 0, 0]).long())

    if is_full_test():
        jit = torch.jit.script(maximal_independent_set_cluster)
        assert cluster1.equal(jit(index, k=0)[1])
        assert cluster2.equal(jit(index, k=1)[1])
        assert cluster3.equal(jit(index, k=2)[1])


def test_kmis_pooling_repr():
    for k in range(3):
        pool1 = KMISPooling(in_channels=16, k=k, scorer='linear')
        assert str(pool1) == f'KMISPooling(in_channels=16, k={k})'
        pool2 = KMISPooling(k=k, scorer='random')
        assert str(pool2) == f'KMISPooling(k={k})'


def _custom_scorer(x, *args, **kwargs):
    return x.norm(dim=-1, keepdim=True)


@pytest.mark.parametrize('scorer', KMISPooling._scorers | {_custom_scorer})
@pytest.mark.parametrize('score_heuristic', KMISPooling._heuristics)
@pytest.mark.parametrize('score_passthrough',
                         KMISPooling._passthroughs - {None})
@pytest.mark.parametrize('aggr_x', ['mean', MaxAggregation(), None])
def test_kmis_pooling(scorer, score_passthrough, score_heuristic, aggr_x):
    index = torch.tensor([[0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 4, 4],
                          [1, 4, 0, 2, 4, 1, 3, 2, 4, 0, 1, 3]]).long()

    num_nodes = 5
    in_channels = 16
    x = torch.rand(num_nodes, in_channels)

    if WITH_TORCH_SPARSE:
        pool1 = KMISPooling(in_channels, k=0, scorer=scorer,
                            score_heuristic=score_heuristic,
                            score_passthrough=score_passthrough, aggr_x=aggr_x)
        out1 = pool1(x, index)
        assert out1[0].shape == x.shape
        assert out1[1].equal(index)

        pool2 = KMISPooling(in_channels, k=1, scorer=scorer,
                            score_heuristic=score_heuristic,
                            score_passthrough=score_passthrough, aggr_x=aggr_x)
        out2 = pool2(x, index)
        assert out2[0].size(0) == 2
        assert out2[1].shape == (2, 2)

        pool3 = KMISPooling(in_channels, k=2, scorer=scorer,
                            score_heuristic=score_heuristic,
                            score_passthrough=score_passthrough, aggr_x=aggr_x)
        out3 = pool3(x, index)
        assert out3[0].size(0) == 1
        assert out3[1].shape == (2, 0)
