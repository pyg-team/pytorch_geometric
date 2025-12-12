import torch

from torch_geometric.contrib.nn import SimplE


def test_simple_scoring():
    model = SimplE(num_nodes=5, num_relations=2, hidden_channels=2)

    # Set embeddings manually for deterministic testing
    model.node_emb.weight.data = torch.tensor([
        [1., 2.],
        [3., 4.],
        [5., 6.],
        [1., 1.],
        [2., 2.],
    ])
    model.node_emb_tail.weight.data = torch.tensor([
        [2., 1.],
        [4., 3.],
        [6., 5.],
        [1., 2.],
        [2., 1.],
    ])
    model.rel_emb.weight.data = torch.tensor([
        [1., 1.],
        [2., 2.],
    ])
    model.rel_emb_inv.weight.data = torch.tensor([
        [1., 2.],
        [2., 1.],
    ])

    # Test scoring: (h=1, r=1, t=2)
    # Score 1: ⟨h_1, v_1, t_2⟩ = sum([3,4] * [2,2] * [6,5]) = sum([6,8] * [6,5]) = sum([36,40]) = 76
    # Score 2: ⟨h_2, v_1_inv, t_1⟩ = sum([5,6] * [2,1] * [4,3]) = sum([10,6] * [4,3]) = sum([40,18]) = 58
    # Final: 0.5 * (76 + 58) = 67.0
    
    score = model(
        head_index=torch.tensor([1]),
        rel_type=torch.tensor([1]),
        tail_index=torch.tensor([2]),
    )
    
    # Manual calculation:
    # Score 1: sum([3,4] * [2,2] * [6,5]) = sum([6,8] * [6,5]) = sum([36,40]) = 76
    # Score 2: sum([5,6] * [2,1] * [4,3]) = sum([10,6] * [4,3]) = sum([40,18]) = 58
    # Final: 0.5 * (76 + 58) = 67.0
    expected_score = 67.0
    assert torch.allclose(score, torch.tensor([expected_score]))


def test_simple():
    model = SimplE(num_nodes=10, num_relations=5, hidden_channels=32)
    assert str(model) == 'SimplE(10, num_relations=5, hidden_channels=32)'

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

