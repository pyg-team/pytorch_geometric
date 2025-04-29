import os

import torch

from torch_geometric.contrib.nn.bias import GraphAttnSpatialBias
from torch_geometric.contrib.nn.models import GraphTransformer


def test_encoder_speed(benchmark, make_perf_batch):
    # -------- config ----------
    NUM_GRAPHS, NODES, DIM = 32, 128, 64
    BASELINE = float(os.getenv("GT_BENCH_BASE_MS", 40)) / 1000
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -------- setup -----------
    batch = make_perf_batch(NUM_GRAPHS, NODES, DIM, device)
    model = GraphTransformer(
        hidden_dim=DIM,
        num_class=10,
        num_encoder_layers=4,
        use_super_node=True,
    ).to(device)
    model.eval()

    # -------- benchmark --------
    def run():
        with torch.no_grad():
            model(batch)

    benchmark.pedantic(run, iterations=10, rounds=100)

    median = benchmark.stats.stats.median
    assert median <= BASELINE * 1.10, (
        f"Encoder forward too slow: {median*1e3:.1f} ms "
        f">(baseline {BASELINE*1e3:.1f} ms ×1.10)"
    )


def test_encoder_with_bias_speed(benchmark, make_perf_batch_with_spatial):
    # -------- config ----------
    NUM_GRAPHS, NODES, DIM = 32, 128, 64
    NUM_SPATIAL, NUM_HEADS = 16, 4
    BASELINE = float(os.getenv("GT_BENCH_BASE_BIAS_MS", 45)) / 1000
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -------- setup -----------
    batch = make_perf_batch_with_spatial(
        NUM_GRAPHS, NODES, DIM, NUM_SPATIAL, device
    )
    model = GraphTransformer(
        hidden_dim=DIM,
        num_class=10,
        num_encoder_layers=4,
        use_super_node=True,
        attn_bias_providers=[
            GraphAttnSpatialBias(
                num_heads=NUM_HEADS,
                num_spatial=NUM_SPATIAL,
                use_super_node=True,
            )
        ]
    ).to(device)
    model.eval()

    # -------- benchmark --------
    def run():
        with torch.no_grad():
            model(batch)

    benchmark.pedantic(run, iterations=10, rounds=100)

    median = benchmark.stats.stats.median
    assert median <= BASELINE * 1.10, (
        f"Encoder+bias forward too slow: {median*1e3:.1f} ms "
        f">(baseline {BASELINE*1e3:.1f} ms ×1.10)"
    )
