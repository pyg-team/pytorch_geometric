import os

import pytest
import torch
import torch.nn as nn

from torch_geometric.contrib.nn.bias import (
    GraphAttnEdgeBias,
    GraphAttnHopBias,
    GraphAttnSpatialBias,
)
from torch_geometric.contrib.nn.models import GraphTransformer
from torch_geometric.contrib.nn.positional import (
    DegreeEncoder,
    EigEncoder,
    SVDEncoder,
)
from torch_geometric.nn import GCNConv


def test_encoder_speed(benchmark, make_perf_batch):
    # -------- config ----------
    NUM_GRAPHS, NODES, HIDDEN_DIM = 32, 128, 64
    BASELINE = float(os.getenv("GT_BENCH_BASE_MS", 40)) / 1000
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -------- setup -----------
    batch = make_perf_batch(NUM_GRAPHS, NODES, HIDDEN_DIM, device)
    model = GraphTransformer(
        hidden_dim=HIDDEN_DIM, num_class=10, encoder_cfg={
            "num_encoder_layers": 4,
            "use_super_node": True,
        }).to(device)
    model.eval()

    # -------- benchmark --------
    def run():
        with torch.no_grad():
            model(batch)

    benchmark.pedantic(run, iterations=10, rounds=100)

    median = benchmark.stats.stats.median
    assert median <= BASELINE * 1.10, (
        f"Encoder forward too slow: {median*1e3:.1f} ms "
        f">(baseline {BASELINE*1e3:.1f} ms ×1.10)")


def test_encoder_with_bias_speed(benchmark, make_perf_batch_with_spatial):
    # -------- config ----------
    NUM_GRAPHS, NODES, HIDDEN_DIM = 32, 128, 64
    NUM_SPATIAL, NUM_HEADS = 16, 4
    BASELINE = float(os.getenv("GT_BENCH_BASE_BIAS_MS", 45)) / 1000
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -------- setup -----------
    batch = make_perf_batch_with_spatial(NUM_GRAPHS, NODES, HIDDEN_DIM,
                                         NUM_SPATIAL, device)
    model = GraphTransformer(
        hidden_dim=HIDDEN_DIM, num_class=10, encoder_cfg={
            "num_encoder_layers":
            4,
            "use_super_node":
            True,
            "attn_bias_providers": [
                GraphAttnSpatialBias(
                    num_heads=NUM_HEADS,
                    num_spatial=NUM_SPATIAL,
                    use_super_node=True,
                )
            ]
        }).to(device)
    model.eval()

    # -------- benchmark --------
    def run():
        with torch.no_grad():
            model(batch)

    benchmark.pedantic(run, iterations=10, rounds=100)

    median = benchmark.stats.stats.median
    assert median <= BASELINE * 1.10, (
        f"Encoder+bias forward too slow: {median*1e3:.1f} ms "
        f">(baseline {BASELINE*1e3:.1f} ms ×1.10)")


@pytest.mark.parametrize(
    "gnn_position",
    ["pre", "post", "parallel"],
    ids=["pre", "post", "parallel"],
)
def test_encoder_full_feature_speed(
    benchmark,
    full_feature_batch,
    structural_config,
    positional_config,
    gnn_position,
):
    # ─── constants ───────────────────────────────────────────────────────────
    HIDDEN_DIM = structural_config["feat_dim"]
    NUM_SPATIAL = positional_config["num_spatial"]
    NUM_EDGES = positional_config["num_edges"]
    NUM_HOPS = positional_config["num_hops"]
    MAX_DEGREE = positional_config["max_degree"]
    NUM_EIGVEC = positional_config["num_eigvec"]
    NUM_SING_VEC = positional_config["num_sing_vec"]
    NUM_HEADS = 4
    NUM_LAYERS = 4
    NUM_CLASSES = 10

    BASELINE = float(os.getenv("GT_BENCH_BASE_FULL_MS", 60)) / 1000
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ─── build a batch with spatial info ───────────────────────────────────
    batch = full_feature_batch(device)
    conv = GCNConv(HIDDEN_DIM, HIDDEN_DIM).to(device)

    def gcn_hook(data, x):
        return conv(x, data.edge_index)

    model = GraphTransformer(
        hidden_dim=HIDDEN_DIM, num_class=NUM_CLASSES, encoder_cfg={
            "num_encoder_layers":
            NUM_LAYERS,
            "use_super_node":
            True,
            "num_heads":
            NUM_HEADS,
            "node_feature_encoder":
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            "positional_encoders": [
                DegreeEncoder(MAX_DEGREE, MAX_DEGREE, HIDDEN_DIM),
                EigEncoder(num_eigvec=NUM_EIGVEC, hidden_dim=HIDDEN_DIM),
                SVDEncoder(r=NUM_SING_VEC, hidden_dim=HIDDEN_DIM),
            ],
            "attn_bias_providers": [
                GraphAttnSpatialBias(num_heads=NUM_HEADS,
                                     num_spatial=NUM_SPATIAL,
                                     use_super_node=True),
                GraphAttnEdgeBias(num_heads=NUM_HEADS, num_edges=NUM_EDGES,
                                  use_super_node=True),
                GraphAttnHopBias(num_heads=NUM_HEADS, num_hops=NUM_HOPS,
                                 use_super_node=True),
            ],
        }, gnn_cfg={
            "gnn_block": gcn_hook,
            "gnn_position": gnn_position,
        }).to(device).eval()

    # ─── benchmark ───────────────────────────────────────────────────────────
    def run():
        with torch.no_grad():
            model(batch)

    benchmark.pedantic(run, iterations=10, rounds=100)
    median = benchmark.stats.stats.median
    assert median <= BASELINE * 1.10, (
        f"Full‐feature ({gnn_position}) too slow: "
        f"{median*1e3:.1f} ms > baseline {BASELINE*1e3:.1f} ms ×1.10")
