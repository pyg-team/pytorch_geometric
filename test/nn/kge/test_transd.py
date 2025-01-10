import pytest
import torch

from torch_geometric.nn import TransD


@pytest.mark.parametrize('channels_node_rel', [(16, 32), {32, 16}])
@pytest.mark.parametrize('bern', [False, True])
def test_transd(channels_node_rel, bern):
    channels_node, channels_rel = channels_node_rel
    model = TransD(num_nodes=10, num_relations=5,
                   hidden_channels_node=channels_node,
                   hidden_channels_rel=channels_rel, bern=bern)
    assert str(model) == ('TransD(10, num_relations=5,'
                          f' hidden_channels_node={channels_node},'
                          f' hidden_channels_rel={channels_rel})')

    head_index = torch.tensor([0, 2, 4, 6, 8])
    rel_type = torch.tensor([0, 1, 2, 3, 4])
    tail_index = torch.tensor([1, 3, 5, 7, 9])

    loader = model.loader(head_index, rel_type, tail_index, batch_size=5)
    for h, r, t in loader:
        out = model(h, r, t)
        assert out.size() == (5, )

        loss = model.loss(h, r, t)
        assert loss >= 0.

        mean_rank, mrr, hits = model.test(h, r, t, batch_size=5, log=False)
        assert 0 <= mean_rank <= 10
        assert 0 < mrr <= 1
        assert hits == 1.0
