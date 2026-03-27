import pytest
import torch
from torch_geometric.nn.pool import SPAPooling


@pytest.fixture()
def simple_graph():
    """Single graph: 10 nodes, 16 features, random edges."""
    torch.manual_seed(0)
    x = torch.randn(10, 16)
    edge_index = torch.randint(0, 10, (2, 30))
    batch = torch.zeros(10, dtype=torch.long)
    return x, edge_index, batch


@pytest.fixture()
def batched_graph():
    """Batch of 2 graphs: 6 + 8 nodes."""
    torch.manual_seed(1)
    x = torch.randn(14, 16)
    # graph 0: nodes 0-5, graph 1: nodes 6-13
    ei0 = torch.randint(0, 6, (2, 15))
    ei1 = torch.randint(6, 14, (2, 20))
    edge_index = torch.cat([ei0, ei1], dim=1)
    batch = torch.cat([torch.zeros(6), torch.ones(8)]).long()
    return x, edge_index, batch


# ── __repr__ ───────────────────────────────────────────────────────────────


def test_repr():
    pool = SPAPooling(ratio=0.5, n_feature=16)
    assert "SPAPooling" in repr(pool)
    assert "n_feature=16" in repr(pool)
    assert "ratio=0.5" in repr(pool)


# ── reset_parameters ────────────────────────────────────────────────────────


def test_reset_parameters():
    pool = SPAPooling(ratio=0.5, n_feature=16)
    pool.reset_parameters()  # should not raise


# ── forward: output shapes ──────────────────────────────────────────────────


def test_forward_shapes_single_graph(simple_graph):
    x, edge_index, batch = simple_graph
    pool = SPAPooling(ratio=0.5, n_feature=16, no=8)
    O, ei_pool, ea, b_pool, perm, score, loss = pool(x, batch, edge_index)

    k = perm.size(0)
    assert O.size() == (k, 16), "Pooled features must have shape (K, F)"
    assert ei_pool.size(0) == 2, "Pooled edge_index must have 2 rows"
    assert b_pool.size(0) == k, "Pooled batch must have K entries"
    assert score.size(0) == k, "Score vector must have K entries"
    assert ea is None, "Edge weight placeholder must be None"


def test_forward_shapes_batched(batched_graph):
    x, edge_index, batch = batched_graph
    pool = SPAPooling(ratio=0.5, n_feature=16, no=8)
    O, ei_pool, _, b_pool, perm, score, loss = pool(x, batch, edge_index)

    k = perm.size(0)
    assert O.size() == (k, 16)
    assert b_pool.size(0) == k
    # representatives must stay within their original graph
    assert (b_pool[b_pool == 0].numel() + b_pool[b_pool == 1].numel()) == k


# ── selection modes ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("select", ["topk", "sagpool"])
def test_select_modes(simple_graph, select):
    x, edge_index, batch = simple_graph
    pool = SPAPooling(ratio=0.5, n_feature=16, select=select, no=8)
    O, ei_pool, _, b_pool, perm, score, loss = pool(x, batch, edge_index)
    assert O.size(1) == 16


def test_select_invalid():
    with pytest.raises(NotImplementedError):
        SPAPooling(ratio=0.5, n_feature=16, select="unknown")


# ── association modes ───────────────────────────────────────────────────────


@pytest.mark.parametrize("asso", ["scalar", "cosine", "attn"])
def test_association_modes(simple_graph, asso):
    x, edge_index, batch = simple_graph
    pool = SPAPooling(ratio=0.5, n_feature=16, asso=asso, no=8)
    O, *_ = pool(x, batch, edge_index)
    assert O.size(1) == 16


# ── loss families ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("loss_fn", ["diffpool", "mincut", "dmon"])
def test_loss_families(simple_graph, loss_fn):
    x, edge_index, batch = simple_graph
    pool = SPAPooling(ratio=0.5, n_feature=16, loss=loss_fn, no=8)
    *_, loss = pool(x, batch, edge_index)
    assert isinstance(loss, dict)
    assert len(loss) > 0
    for v in loss.values():
        assert torch.isfinite(v), f"Loss '{loss_fn}' returned a non-finite value"


def test_manual_loss(simple_graph):
    x, edge_index, batch = simple_graph
    pool = SPAPooling(
        ratio=0.5, n_feature=16, loss="manual", regs=["link", "entropy"], no=8
    )
    *_, loss = pool(x, batch, edge_index)
    assert "link" in loss and "entropy" in loss


# ── min_score mode ──────────────────────────────────────────────────────────


def test_min_score_mode(simple_graph):
    x, edge_index, batch = simple_graph
    pool = SPAPooling(ratio=0.5, n_feature=16, min_score=0.5, no=8)
    O, ei_pool, _, b_pool, perm, score, loss = pool(x, batch, edge_index)
    assert O.size(1) == 16


# ── gradients flow ──────────────────────────────────────────────────────────


def test_gradients(simple_graph):
    x, edge_index, batch = simple_graph
    x = x.requires_grad_(True)
    pool = SPAPooling(ratio=0.5, n_feature=16, no=8)
    O, *_, loss = pool(x, batch, edge_index)
    total_loss = O.sum() + sum(loss.values())
    total_loss.backward()
    assert x.grad is not None, "Gradients must flow back to input features"
