import torch

from torch_geometric.nn import TransR


def test_transr():
    model = TransR(num_nodes=10, num_relations=5, hidden_channels=32)
    assert str(model) == 'TransR(10, num_relations=5, hidden_channels=32)'

    # Create test indices
    head_index = torch.tensor([0, 2, 4, 6, 8])
    rel_type = torch.tensor([0, 1, 2, 3, 4])
    tail_index = torch.tensor([1, 3, 5, 7, 9])

    # Test data loading
    loader = model.loader(head_index, rel_type, tail_index, batch_size=5)
    for h, r, t in loader:
        # Test forward pass
        out = model(h, r, t)
        assert out.size() == (5,)

        # Test loss computation
        loss = model.loss(h, r, t)
        assert loss >= 0.

        # Test evaluation metrics
        mean_rank, mrr, hits = model.test(h, r, t, batch_size=5, log=False)
        assert 0 <= mean_rank <= 10
        assert 0 < mrr <= 1
        assert hits == 1.0

    # Test projection
    for h, r, t in loader:
        head = model.node_emb(h)
        head_proj = model._project(head, r)
        assert head_proj.size() == (5, model.hidden_channels)
        
    # Test embeddings normalization
    assert torch.allclose(
        model.rel_emb.weight.norm(p=model.p_norm, dim=-1),
        torch.ones(model.num_relations),
        atol=1e-5
    )