import torch

from torch_geometric.nn import ConvE


def test_conve():
    model = ConvE(num_nodes=10, num_relations=5, hidden_channels=50)
    assert str(model) == 'ConvE(10, num_relations=5, hidden_channels=50)'

    # Test different initialization parameters
    model = ConvE(
        num_nodes=10,
        num_relations=5,
        hidden_channels=50,
        embedding_height=5,
        input_drop=0.1,
        hidden_drop=0.2,
        feat_drop=0.3,
        use_bias=False,
    )

    # Test shapes
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