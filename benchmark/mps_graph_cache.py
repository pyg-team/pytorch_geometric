#!/usr/bin/env python3
"""Benchmark: MPS graph cache policy impact on ZINC molecular GNN training.

PyG's Batch.from_data_list() concatenates variable-size graphs so the batch
shape key is (sum_nodes, sum_edges) -- a 2-D combinatorial space. With
shuffled DataLoaders every batch is unique: P(repeat) ~ 1/C(N, BS) ~ 0.
MPSGraph caches a compiled graph for each shape indefinitely. Over a 100-epoch
run on ZINC (10k molecules, BS=32): ~31k unique shapes, ~1.7 GB leaked cache.

Each strategy runs in an isolated subprocess for clean RSS measurement
(macOS ru_maxrss is monotonic; psutil current RSS requires a fresh process).

Usage:
    PYTHONPATH=. python3 benchmark/mps_graph_cache.py

Requires: torch >= 2.13 (pytorch/pytorch#182648), torch_geometric, psutil, MPS.
"""
import argparse
import json
import math
import os
import random
import statistics
import subprocess
import sys
import tempfile
import time

import psutil
import torch
import torch.nn.functional as F
from torch import nn

assert torch.backends.mps.is_available(), "MPS not available"
assert hasattr(torch.mps, 'freeze_graph_cache'), (
    "freeze_graph_cache requires PyTorch >= 2.13 (pytorch/pytorch#182648)")

DEVICE = torch.device("mps")
ITERS  = 200
WARMUP = 5


def _cur_rss_mb():
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024


def _make_workload(quiet=False):
    from torch_geometric.data import Batch
    from torch_geometric.datasets import ZINC
    from torch_geometric.nn import GCNConv, global_mean_pool

    root = os.path.join(tempfile.gettempdir(), "pyg_data")
    dataset = ZINC(root=root, subset=True, split="train")
    bs = 32

    node_counts = [dataset[i].num_nodes for i in range(len(dataset))]
    edge_counts = [dataset[i].num_edges for i in range(len(dataset))]
    std_n = statistics.stdev(node_counts)
    std_e = statistics.stdev(edge_counts)
    eff_2d = math.sqrt(bs) * std_n * math.sqrt(bs) * std_e * 2 * math.pi
    freeze_at = min(ITERS // 4, max(5, int(math.sqrt(eff_2d))))

    if not quiet:
        print(f"  ZINC: {len(dataset)} molecules  "
              f"std_nodes={std_n:.1f}  std_edges={std_e:.1f}  "
              f"eff_2d={eff_2d:.0f}  freeze_at={freeze_at}")

    in_dim = dataset[0].x.shape[1]

    class MolGNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = GCNConv(in_dim, 64)
            self.conv2 = GCNConv(64, 64)
            self.lin   = nn.Linear(64, 1)

        def forward(self, x, edge_index, batch):
            x = F.relu(self.conv1(x, edge_index))
            x = F.relu(self.conv2(x, edge_index))
            return self.lin(global_mean_pool(x, batch))

    model     = MolGNN().to(DEVICE).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    def step_fn(i):
        rng = random.Random(i)
        idx = list(range(len(dataset)))
        rng.shuffle(idx)
        batch = Batch.from_data_list(
            [dataset[idx[j]] for j in range(bs)]
        ).to(DEVICE)
        target = torch.randn(bs, 1, device=DEVICE)
        optimizer.zero_grad()
        out  = model(batch.x.float(), batch.edge_index, batch.batch)
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()

    return step_fn, eff_2d, freeze_at, len(dataset) // bs


# -- subprocess runner --------------------------------------------------------

def run_strategy(strat_name, freeze_at):
    if strat_name == "never":
        torch.mps.set_graph_cache_policy("never")
    else:
        torch.mps.set_graph_cache_policy("always")

    step_fn, eff_2d, _, _ = _make_workload(quiet=True)
    torch.mps.empty_cache()
    torch.mps.clear_graph_cache()
    torch.mps.synchronize()

    rss_pre = _cur_rss_mb()

    for i in range(WARMUP):
        step_fn(i)
    torch.mps.synchronize()

    rss_post_warmup = _cur_rss_mb()

    times = []
    drvs  = []
    iter_peak_rss = rss_pre
    rss_at_freeze = None

    for i in range(ITERS):
        t0 = time.perf_counter()
        step_fn(WARMUP + i)

        if strat_name == "clear_per_iter":
            iter_peak_rss = max(iter_peak_rss, _cur_rss_mb())
            torch.mps.clear_graph_cache()
        elif strat_name == "freeze_after_warmup" and i == freeze_at:
            rss_at_freeze = _cur_rss_mb()
            torch.mps.freeze_graph_cache()

        torch.mps.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
        drvs.append(torch.mps.driver_allocated_memory())

    torch.mps.synchronize()
    rss_final = _cur_rss_mb()

    n = len(drvs)
    xm = (n - 1) / 2.0
    denom = max(1, sum((j - xm) ** 2 for j in range(n)))
    slope_kbpi = sum(
        (j - xm) * (drvs[j] - drvs[0]) for j in range(n)
    ) / denom / 1024

    return {
        "strat":        strat_name,
        "rss_delta":    rss_final - rss_pre,
        "warmup_swell": rss_post_warmup - rss_pre,
        "freeze_swell": (rss_at_freeze - rss_pre) if rss_at_freeze is not None else None,
        "iter_peak":    (iter_peak_rss - rss_pre) if strat_name == "clear_per_iter" else None,
        "mean_ms":      statistics.mean(times),
        "std_ms":       statistics.stdev(times) if len(times) > 1 else 0.0,
        "slope_kbpi":   slope_kbpi,
        "eff_2d":       eff_2d,
    }


# -- orchestrator -------------------------------------------------------------

def main_benchmark():
    _, eff_2d, freeze_at, iters_per_epoch = _make_workload()
    strategies = ["always", "freeze_after_warmup", "clear_per_iter", "never"]

    print()
    print(f"PyTorch {torch.__version__}  |  ZINC MPS graph cache benchmark")
    print(f"  warmup={WARMUP}  freeze_at={freeze_at}  iters={ITERS}  (subprocess-isolated)")
    print()
    print(f"  {'Strategy':<22} {'ΔRSS MB':>9}  {'ms/iter':>9}  {'+-':>6}  note")
    print(f"  {'-'*22} {'-'*9}  {'-'*9}  {'-'*6}  ------")

    always_ms = None
    always_slope = None

    for strat_name in strategies:
        sys.stdout.write(f"  {strat_name:<22} running...\r")
        sys.stdout.flush()

        result = subprocess.run(
            [sys.executable, __file__, "--strategy", strat_name,
             "--freeze-at", str(freeze_at)],
            capture_output=True, text=True,
            env=os.environ.copy(),
        )

        if result.returncode != 0:
            print(f"  {strat_name:<22} FAILED: {result.stderr[-200:]}")
            continue

        metrics = None
        for line in result.stdout.splitlines():
            if line.startswith("RESULT:"):
                metrics = json.loads(line[7:])
                break

        if metrics is None:
            print(f"  {strat_name:<22} no result")
            continue

        if always_ms is None:
            always_ms = metrics["mean_ms"]
            always_slope = metrics["slope_kbpi"]
            note = "(baseline, unbounded)"
        else:
            note = f"({metrics['mean_ms'] / always_ms:.2f}x, bounded)"

        print(f"  {strat_name:<22} {metrics['rss_delta']:>+9.1f}  "
              f"{metrics['mean_ms']:>9.1f}  {metrics['std_ms']:>6.1f}  {note}")

        if strat_name == "freeze_after_warmup" and metrics["freeze_swell"] is not None:
            print(f"    warmup loop: +{metrics['warmup_swell']:.1f} MB  |  "
                  f"at freeze point: +{metrics['freeze_swell']:.1f} MB")

        if strat_name == "clear_per_iter" and metrics["iter_peak"] is not None:
            print(f"    peak/iter before clear: +{metrics['iter_peak']:.1f} MB  "
                  f"(net: {metrics['rss_delta']:+.1f} MB)")

    if always_slope and always_slope > 0:
        epochs = 100
        extrap_gb = always_slope * iters_per_epoch * epochs / 1024 / 1024
        print()
        print(f"  always: {always_slope:.1f} KB/iter x {iters_per_epoch} iters/epoch "
              f"x {epochs} epochs = {extrap_gb:.1f} GB extrapolated cache growth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default=None,
                        choices=["always", "freeze_after_warmup",
                                 "clear_per_iter", "never"])
    parser.add_argument("--freeze-at", type=int, default=50)
    args = parser.parse_args()

    if args.strategy:
        metrics = run_strategy(args.strategy, args.freeze_at)
        print("RESULT:" + json.dumps(metrics))
    else:
        main_benchmark()
