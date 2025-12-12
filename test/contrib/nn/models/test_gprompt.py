from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from torch_geometric.contrib.nn.models.gprompt import GraphAdapter


def test_graph_adapter_forward_shapes():
    N, M, H, Z, V = 4, 2, 8, 8, 5
    masked_edges = 2

    z = torch.randn(N, Z)
    h_mask = torch.randn(M, H)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])
    node_ids = torch.tensor([1, 2])
    predictor = nn.Linear(H, V)

    adapter = GraphAdapter(hidden_channels=H, node_channels=Z)
    logits, masked_idx = adapter(
        h_mask=h_mask,
        z=z,
        edge_index=edge_index,
        node_ids=node_ids,
        predictor=predictor,
    )

    assert logits.shape == (masked_edges, V)
    assert masked_idx.shape == (masked_edges, )


def test_graph_adapter_loss_equivalence():
    # Verify that the loss function matches manual average of
    # cross-entropy for a single masked token.
    edge_logits = torch.tensor([[1.0, 0.0, -1.0], [0.5, 0.5, -1.0]])
    masked_idx = torch.tensor([0, 0])
    target_ids = torch.tensor([1])

    loss = GraphAdapter.loss(
        edge_logits=edge_logits,
        masked_idx=masked_idx,
        target_ids=target_ids,
    )

    log_probs = F.log_softmax(edge_logits, dim=-1)
    gt = torch.tensor([1, 1])
    manual = F.nll_loss(log_probs, gt, reduction="mean")

    assert torch.allclose(loss, manual)


def test_graph_adapter_reduces_to_predictor_when_gate_is_large():
    N, M, H, Z, V = 3, 1, 4, 4, 6

    torch.manual_seed(0)
    z = torch.zeros(N, Z)
    h_mask = torch.randn(M, H)
    edge_index = torch.tensor([[1, 2],
                               [0, 0]])  # two neighbors -> same center node 0
    node_ids = torch.tensor([0])
    predictor = nn.Linear(H, V, bias=False)

    adapter = GraphAdapter(hidden_channels=H, node_channels=Z)

    # Make q_proj and k_proj output large positive values for node 0
    # to push a_ij -> 1
    with torch.no_grad():
        adapter.q_proj.weight.fill_(1.0)
        adapter.q_proj.bias.fill_(100.0)
        adapter.k_proj.weight.fill_(1.0)
        adapter.k_proj.bias.fill_(100.0)

        # Make g_mlp ~ 0 so the second term vanishes:
        for p in adapter.g_mlp.parameters():
            p.zero_()

    logits_pooled, _ = adapter(
        h_mask=h_mask,
        z=z,
        edge_index=edge_index,
        node_ids=node_ids,
        predictor=predictor,
    )

    # Direct predictor on h_mask:
    direct_logits = predictor(h_mask)

    # They won't be exactly identical (sigmoid not exactly 1), but
    # should be close.
    assert torch.allclose(logits_pooled, direct_logits, atol=1e-3, rtol=1e-3)


def test_pool_outputs_log_vs_prob():
    edge_logits = torch.tensor([[1.0, 0.0],
                                [3.0, 0.0]])  # two edges, same masked token
    masked_idx = torch.tensor([0, 0])
    M = 1
    # Arithmetic mean of log probs (log of geometric mean of probs)
    log_pooled = GraphAdapter.pool_outputs(edge_logits, masked_idx, M,
                                           use_log_probs=True)
    assert log_pooled.shape == (M, 2)
    per_edge_log_probs = F.log_softmax(edge_logits, dim=-1)
    manual_log_mean = per_edge_log_probs.mean(dim=0)
    assert torch.allclose(log_pooled[0], manual_log_mean, atol=1e-6)
    # Arithmetic mean of probs
    prob_pooled = GraphAdapter.pool_outputs(edge_logits, masked_idx, M,
                                            use_log_probs=False)
    assert prob_pooled.shape == (M, 2)
    per_edge_probs = F.softmax(edge_logits, dim=-1)
    manual_prob_mean = per_edge_probs.mean(dim=0)
    assert torch.allclose(prob_pooled[0], manual_prob_mean, atol=1e-6)


def test_multiple_masked_tokens_same_node():
    # Node 0 has two masked tokens, and two neighbors.
    N, H, V = 3, 4, 7
    z = torch.randn(N, H)
    h_mask = torch.randn(2, H)  # two tokens on node 0
    edge_index = torch.tensor([[1, 2], [0, 0]])  # 1->0, 2->0
    node_ids = torch.tensor([0, 0])
    predictor = nn.Linear(H, V)

    adapter = GraphAdapter(hidden_channels=H, node_channels=H)
    edge_logits, masked_idx = adapter(
        h_mask=h_mask,
        z=z,
        edge_index=edge_index,
        node_ids=node_ids,
        predictor=predictor,
    )
    assert edge_logits.shape[0] == 4
    assert masked_idx.tolist().count(0) == 2
    assert masked_idx.tolist().count(1) == 2


def test_forward_no_neighbors():
    N, M, H, Z, V = 4, 2, 8, 8, 5
    h_mask = torch.randn(M, H)
    z = torch.randn(N, Z)
    # No edges with dest being 0 or 1
    edge_index = torch.tensor(
        [
            [1, 0],
            [2, 3],
        ],
        dtype=torch.long,
    )
    node_ids = torch.tensor([0, 1], dtype=torch.long)
    predictor = nn.Linear(H, V)
    adapter = GraphAdapter(hidden_channels=H, node_channels=Z)
    edge_logits, masked_idx = adapter(
        h_mask=h_mask,
        z=z,
        edge_index=edge_index,
        node_ids=node_ids,
        predictor=predictor,
    )
    assert edge_logits.shape == (0, V)
    assert masked_idx.numel() == 0
    assert masked_idx.dtype == torch.long


def test_loss_no_neighbors():
    V = 5
    edge_logits = torch.empty((0, V))
    masked_idx = torch.empty((0, ), dtype=torch.long)
    target_ids = torch.tensor([1, 2])

    loss = GraphAdapter.loss(
        edge_logits=edge_logits,
        masked_idx=masked_idx,
        target_ids=target_ids,
    )
    assert loss.item() == 0.0


def test_backward_computes_gradients():
    N, M, H, Z, V = 4, 2, 8, 8, 5
    z = torch.randn(N, Z, requires_grad=False)
    h_mask = torch.randn(M, H, requires_grad=False)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])
    node_ids = torch.tensor([1, 2])
    target_ids = torch.tensor([1, 2])
    predictor = nn.Linear(H, V)

    adapter = GraphAdapter(hidden_channels=H, node_channels=Z)
    edge_logits, masked_idx = adapter(
        h_mask=h_mask,
        z=z,
        edge_index=edge_index,
        node_ids=node_ids,
        predictor=predictor,
    )
    loss = adapter.loss(edge_logits, masked_idx, target_ids)
    loss.backward()

    grads = [p.grad for p in adapter.parameters()]
    assert any(g is not None and g.abs().sum() > 0 for g in grads)
