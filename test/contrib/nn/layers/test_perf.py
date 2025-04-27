import os

import pytest
import torch

from torch_geometric.contrib.nn.models import GraphTransformer
from torch_geometric.data import Batch, Data


def _make_batch(num_graphs, nodes_per_graph, feat_dim, device):
    data = []
    for _ in range(num_graphs):
        x = torch.randn(nodes_per_graph, feat_dim, device=device)
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        data.append(Data(x=x, edge_index=edge_index))
    return Batch.from_data_list(data)


@pytest.mark.skipif(
    not torch.cuda.is_available() and os.getenv("CI_LOCAL") is None,
    reason="Perf guard only enforced on CI_GPU or when CI_LOCAL=1",
)
def test_encoder_speed(benchmark):
    # -------- config ----------
    NUM_GRAPHS, NODES, DIM = 32, 128, 64
    BASELINE = float(os.getenv("GT_BENCH_BASE_MS", 40)) / 1000  # seconds
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -------- setup -----------
    batch = _make_batch(NUM_GRAPHS, NODES, DIM, device)
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
