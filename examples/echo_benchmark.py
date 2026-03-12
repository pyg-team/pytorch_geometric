#!/usr/bin/env python3

"""Minimal ADGN-style regression example for :class:`ECHOBenchmark`.

This example is intentionally small and uses only PyTorch + PyG.
It follows the upstream ECHO reference code in the parts that matter for a
minimal reproduction:

- `log10(MSE)` is used as the training/validation loss.
- `sssp`, `ecc`, and `charge` are treated as node-level regression tasks.
- `diam` and `energy` are treated as graph-level regression tasks.
- `energy` is evaluated in the linear domain via `10**pred` and `10**target`.

Usage:
    python examples/echo_benchmark.py --task sssp
    python examples/echo_benchmark.py --task diam --epochs 20
    python examples/echo_benchmark.py --task energy --batch_size 16
"""

from __future__ import annotations

import argparse
import copy
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from torch_geometric.data import Data
from torch_geometric.datasets import ECHOBenchmark
from torch_geometric.loader import DataLoader
from torch_geometric.nn import (
    AntiSymmetricConv,
    MessagePassing,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)


NODE_LEVEL_TASKS = {"sssp", "ecc", "charge"}
GRAPH_LEVEL_TASKS = {"diam", "energy"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_node_level_task(task: str) -> bool:
    return task in NODE_LEVEL_TASKS


def ensure_2d(value) -> torch.Tensor:
    if isinstance(value, (list, tuple)):
        value = np.asarray(value)
    value = torch.as_tensor(value)
    if value.dim() == 0:
        return value.view(1, 1)
    if value.dim() == 1:
        return value.view(-1, 1)
    return value


def prepare_data(data: Data) -> Data:
    data.edge_index = torch.as_tensor(np.asarray(data.edge_index)).long()
    data.x = ensure_2d(data.x).float()

    edge_attr = getattr(data, "edge_attr", None)
    if edge_attr is not None:
        data.edge_attr = ensure_2d(edge_attr).float()

    data.y = ensure_2d(data.y).float()
    return data


def infer_input_dims(sample: Data) -> tuple[int, int]:
    sample = prepare_data(sample)
    input_dim = sample.x.size(-1)
    edge_dim = 0 if getattr(sample, "edge_attr", None) is None else sample.edge_attr.size(-1)
    return input_dim, edge_dim


def infer_output_dim(sample: Data) -> int:
    sample = prepare_data(sample)
    return int(sample.y.size(-1))


class SumPhi(MessagePassing):
    """Simple message function used inside :class:`AntiSymmetricConv`."""

    def __init__(self, channels: int, edge_dim: int = 0):
        super().__init__(aggr="add")
        self.lin = nn.Linear(channels, channels, bias=False)
        self.edge_lin = nn.Linear(edge_dim, channels, bias=False) if edge_dim > 0 else None

    def reset_parameters(self) -> None:
        super().reset_parameters()
        self.lin.reset_parameters()
        if self.edge_lin is not None:
            self.edge_lin.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.propagate(edge_index, x=self.lin(x), edge_attr=edge_attr)

    def message(
        self,
        x_j: torch.Tensor,
        edge_attr: torch.Tensor | None,
    ) -> torch.Tensor:
        if edge_attr is None:
            return x_j

        edge_attr = ensure_2d(edge_attr)
        if self.edge_lin is None:
            return edge_attr * x_j

        return x_j + self.edge_lin(edge_attr)


class ADGNRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        edge_dim: int,
        node_level_task: bool,
        epsilon: float,
        gamma: float,
    ) -> None:
        super().__init__()
        self.node_level_task = node_level_task

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.conv = AntiSymmetricConv(
            in_channels=hidden_dim,
            phi=SumPhi(hidden_dim, edge_dim=edge_dim),
            num_iters=num_layers,
            epsilon=epsilon,
            gamma=gamma,
        )

        readout_dim = hidden_dim if node_level_task else hidden_dim * 3
        self.mlp = nn.Sequential(
            nn.Linear(readout_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, data: Data) -> torch.Tensor:
        x = self.input_proj(data.x)
        x = self.conv(x, data.edge_index, edge_attr=getattr(data, "edge_attr", None))

        if self.node_level_task:
            return self.mlp(x).view(-1)

        pooled = torch.cat([
            global_add_pool(x, data.batch),
            global_max_pool(x, data.batch),
            global_mean_pool(x, data.batch),
        ], dim=-1)
        return self.mlp(pooled).view(-1)


def log10_mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.log10(F.mse_loss(pred, target) + 1e-12)


def flatten_targets(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return pred.view(-1), target.view(-1)


def train(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    limit_batches: int | None = None,
) -> float:
    model.train()
    total_loss = 0.0
    num_steps = 0

    for num_steps, batch in enumerate(loader, start=1):
        batch = batch.to(device)
        pred, target = flatten_targets(model(batch), batch.y)

        optimizer.zero_grad(set_to_none=True)
        loss = log10_mse_loss(pred, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if limit_batches is not None and num_steps >= limit_batches:
            break

    return total_loss / max(1, num_steps)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    task: str,
    limit_batches: int | None = None,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_abs_error = 0.0
    total_sq_error = 0.0
    total_examples = 0
    num_steps = 0

    for num_steps, batch in enumerate(loader, start=1):
        batch = batch.to(device)
        pred, target = flatten_targets(model(batch), batch.y)

        total_loss += float(log10_mse_loss(pred, target))

        if task == "energy":
            pred_for_metrics = torch.pow(10.0, pred)
            target_for_metrics = torch.pow(10.0, target)
        else:
            pred_for_metrics = pred
            target_for_metrics = target

        total_abs_error += F.l1_loss(
            pred_for_metrics,
            target_for_metrics,
            reduction="sum",
        ).item()
        total_sq_error += F.mse_loss(
            pred_for_metrics,
            target_for_metrics,
            reduction="sum",
        ).item()
        total_examples += target_for_metrics.numel()

        if limit_batches is not None and num_steps >= limit_batches:
            break

    total_examples = max(1, total_examples)
    return {
        "loss": total_loss / max(1, num_steps),
        "mae": total_abs_error / total_examples,
        "mse": total_sq_error / total_examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Minimal ADGN-style example for ECHOBenchmark.",
    )
    parser.add_argument("--root", type=str, default="data/ECHOBenchmark")
    parser.add_argument(
        "--task",
        type=str,
        default="sssp",
        choices=sorted(NODE_LEVEL_TASKS | GRAPH_LEVEL_TASKS),
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--limit_train_batches",
        type=int,
        default=0,
        help="Use a positive value for a very quick smoke run.",
    )
    parser.add_argument(
        "--limit_eval_batches",
        type=int,
        default=0,
        help="Use a positive value for a very quick smoke run.",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    node_level_task = is_node_level_task(args.task)
    train_limit = args.limit_train_batches or None
    eval_limit = args.limit_eval_batches or None

    train_dataset = ECHOBenchmark(root=args.root, task=args.task, split="train")
    val_dataset = ECHOBenchmark(root=args.root, task=args.task, split="val")
    test_dataset = ECHOBenchmark(root=args.root, task=args.task, split="test")

    train_dataset.transform = prepare_data
    val_dataset.transform = prepare_data
    test_dataset.transform = prepare_data

    sample = train_dataset[0]
    input_dim, edge_dim = infer_input_dims(sample)
    output_dim = infer_output_dim(sample)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = ADGNRegressor(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=output_dim,
        num_layers=args.num_layers,
        edge_dim=edge_dim,
        node_level_task=node_level_task,
        epsilon=args.epsilon,
        gamma=args.gamma,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_state = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")

    print(
        f"task={args.task} node_level={node_level_task} "
        f"train={len(train_dataset)} val={len(val_dataset)} test={len(test_dataset)} "
        f"input_dim={input_dim} edge_dim={edge_dim} output_dim={output_dim} device={device}"
    )

    for epoch in range(1, args.epochs + 1):
        train_loss = train(
            model,
            train_loader,
            optimizer,
            device=device,
            limit_batches=train_limit,
        )
        val_metrics = evaluate(
            model,
            val_loader,
            device=device,
            task=args.task,
            limit_batches=eval_limit,
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_state = copy.deepcopy(model.state_dict())

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_loss:.6f} "
            f"val_loss={val_metrics['loss']:.6f} "
            f"val_mae={val_metrics['mae']:.6f} "
            f"val_mse={val_metrics['mse']:.6f}"
        )

    model.load_state_dict(best_state)
    test_metrics = evaluate(
        model,
        test_loader,
        device=device,
        task=args.task,
        limit_batches=eval_limit,
    )
    print(
        f"best_val_loss={best_val_loss:.6f} "
        f"test_loss={test_metrics['loss']:.6f} "
        f"test_mae={test_metrics['mae']:.6f} "
        f"test_mse={test_metrics['mse']:.6f}"
    )


if __name__ == "__main__":
    main()
