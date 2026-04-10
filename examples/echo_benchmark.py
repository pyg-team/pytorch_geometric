# Minimal example for :class:`ECHOBenchmark`, following the experimental design 
# of the original ECHO Benchmark reference code: https://github.com/Graph-ECHO-Benchmark/ECHO/
# Usage:
#     python examples/echo_benchmark.py --task sssp
#     python examples/echo_benchmark.py --task diam --epochs 20
#     python examples/echo_benchmark.py --task energy --batch_size 16

import argparse
import os
import random

import torch
import torch.nn.functional as F
from torch import nn

from torch_geometric.data import Data
from torch_geometric.datasets import ECHOBenchmark
from torch_geometric.loader import DataLoader
from torch_geometric.nn import (
    GCNConv,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)

# NODE_LEVEL_TASKS = {"sssp", "ecc", "charge"}
# GRAPH_LEVEL_TASKS = {"diam", "energy"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class GCN(nn.Module):
    # A simple GCN model that supports both node-level and graph-level tasks
    # as designed in the original ECHO Benchmark reference code.
    # https://github.com/Graph-ECHO-Benchmark/ECHO/blob/main/models/gnn.py

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int, node_level_task: bool) -> None:
        super().__init__()
        self.node_level_task = node_level_task

        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.convs = nn.ModuleList([
            GCNConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
            ) for _ in range(num_layers)
        ])

        readout_dim = hidden_dim if node_level_task else hidden_dim * 3

        # The readout layer structure from the original ECHO Benchmark reference code
        self.readout = nn.Sequential(nn.Linear(readout_dim, hidden_dim // 2),
                                     nn.LeakyReLU(),
                                     nn.Linear(hidden_dim // 2, output_dim))

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch
        data.edge_attr if hasattr(data, 'edge_attr') else None

        h = self.encoder(x)
        for conv in self.convs:
            h = conv(h, edge_index)

        if not self.node_level_task:
            # Concatenate global add, max, and mean pooling results
            # as in the original ECHO Benchmark reference code
            h = torch.cat([
                global_add_pool(h, batch),
                global_max_pool(h, batch),
                global_mean_pool(h, batch),
            ], dim=1)
        return self.readout(h)


def log10_mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # The original ECHO Benchmark reference code uses log10 of MSE loss.
    return torch.log10(F.mse_loss(pred, target))


def train(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0

    for batch in loader:
        optimizer.zero_grad()

        batch = batch.to(device)
        pred = model(batch)

        loss = log10_mse_loss(pred, batch.y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    task: str,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    total_mse = 0.0

    for batch in loader:
        batch = batch.to(device)
        pred = model(batch)
        target = batch.y

        total_loss += float(log10_mse_loss(pred, target))

        if task == "energy":
            # The ECHO_Benchmark normalizes the target values as the
            # log10(original_graph_energy), so we need to apply the inverse
            # transformation to get the original energy values before computing MAE and MSE
            pred = torch.pow(10.0, pred)
            target = torch.pow(10.0, target)
        elif task in {"sssp", "ecc", "diam"}:
            # The ECHO_Benchmark normalizes the target values for sssp, ecc, and
            # diam by dividing by 40.0 during training
            pred *= 40.0
            target *= 40.0

        total_mae += F.l1_loss(pred, target).item()
        total_mse += F.mse_loss(pred, target).item()

    return {
        "loss": total_loss / len(loader),
        "mae": total_mae / len(loader),
        "mse": total_mse / len(loader),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Minimal example for ECHOBenchmark.", )
    parser.add_argument("--root", type=str, default="data/ECHOBenchmark")
    parser.add_argument(
        "--task",
        type=str,
        default="sssp",
        choices=["sssp", "ecc", "diam", "charge", "energy"],
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset = ECHOBenchmark(root=args.root, task=args.task,
                                  split="train")
    val_dataset = ECHOBenchmark(root=args.root, task=args.task, split="val")
    test_dataset = ECHOBenchmark(root=args.root, task=args.task, split="test")

    input_dim = train_dataset.num_features
    edge_dim = train_dataset.num_edge_features
    output_dim = train_dataset.num_classes
    node_level_task = train_dataset.is_node_level_task

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False)

    model = GCN(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=output_dim,
        num_layers=args.num_layers,
        node_level_task=node_level_task,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val_loss = float("inf")

    print(
        f"task={args.task} node_level={node_level_task} "
        f"num_train_samples={len(train_dataset)} num_val_samples={len(val_dataset)} num_test_samples={len(test_dataset)} "
        f"input_dim={input_dim} edge_dim={edge_dim} output_dim={output_dim} device={device}"
    )

    for epoch in range(1, args.epochs + 1):
        train_loss = train(
            model,
            train_loader,
            optimizer,
            device=device,
        )

        val_metrics = evaluate(
            model,
            val_loader,
            device=device,
            task=args.task,
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save((epoch, model.state_dict()), "best_model.pth")

        print(f"epoch={epoch} "
              f"train_loss={train_loss:.6f} "
              f"val_loss={val_metrics['loss']:.6f} "
              f"val_mae={val_metrics['mae']:.6f} "
              f"val_mse={val_metrics['mse']:.6f}")

    epoch, ckpt = torch.load("best_model.pth")
    model.load_state_dict(ckpt)
    test_metrics = evaluate(
        model,
        test_loader,
        device=device,
        task=args.task,
    )
    print(f"best_epoch={epoch} "
          f"best_val_loss={best_val_loss:.6f} "
          f"test_loss={test_metrics['loss']:.6f} "
          f"test_mae={test_metrics['mae']:.6f} "
          f"test_mse={test_metrics['mse']:.6f}")


if __name__ == "__main__":
    main()
