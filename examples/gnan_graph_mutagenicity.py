"""Training GNAN (graph‐level) on the Mutagenicity dataset.

This reproduces, in simplified form, the experiment from the GNAN paper
(Bechler-Speicher *et al.*, 2024) on the Mutagenicity molecule dataset.

The script demonstrates how to:
1. Load the Mutagenicity dataset from TUDataset.
2. Pre-compute all-pairs shortest-path distances **per graph** and the
   corresponding normalisation matrices required by GNAN.
3. Train the *TensorGNAN* model for graph classification.

Run with:

    python examples/gnan_graph_mutagenicity.py

Graphs in Mutagenicity are small (≈30 nodes), therefore the dense distance
matrix fits comfortably in memory and can be coqmputed on the fly.
"""

from __future__ import annotations

import pickle
import random
from pathlib import Path

import networkx as nx
import torch
from torch import nn
from tqdm import tqdm

from torch_geometric.datasets import TUDataset
from torch_geometric.loader.gnan_dataloader import GNANDataLoader
from torch_geometric.nn.models import TensorGNAN
from torch_geometric.utils import to_networkx


def compute_dist_and_norm(data) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (distance_matrix, normalisation_matrix) for a PyG Data graph."""
    def norm_from_dist(dist: torch.Tensor) -> torch.Tensor:
        N = dist.size(0)
        norm = torch.zeros_like(dist)
        for i in range(N):
            row = dist[i]
            # Consider only *finite* distances when counting
            finite_mask = torch.isfinite(row)
            counts = torch.bincount(row[finite_mask].long(
            )) if finite_mask.any() else torch.tensor([], dtype=torch.long)

            for j in range(N):
                if not torch.isfinite(row[j]):
                    # No path ⇒ normalisation of 1 to avoid division by zero
                    norm[i, j] = 1.0
                else:
                    d = int(row[j].item())
                    norm[i, j] = counts[d] if d < len(counts) else 1.0
        # Safety: ensure no zeros
        norm[norm == 0] = 1.0
        return norm

    g = to_networkx(data, to_undirected=True)
    sp = dict(nx.all_pairs_shortest_path_length(g))

    N = data.num_nodes
    # Initialise with +inf to mark "no path" entries explicitly
    dist = torch.full((N, N), float('inf'), dtype=torch.float)

    # Distance from each node to itself is 0 by definition
    dist.fill_diagonal_(0.0)

    # Fill finite shortest-path lengths returned by NetworkX
    for i, lengths in sp.items():
        for j, d in lengths.items():
            dist[i, j] = float(d)

    # Compute the normalisation matrix; unreachable pairs (inf) get count 1
    norm = norm_from_dist(dist)
    return dist, norm


class PreprocessDistances:
    """PyG Transform that adds GNAN distance attributes to each graph."""
    def __call__(self, data):  # noqa: D401
        dist, norm = compute_dist_and_norm(data)
        data.node_distances = dist
        data.normalization_matrix = norm
        return data


# -----------------------------------------------------------------------------


def seed_everything(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------------------------------------------------------


def main():
    seed_everything()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    root = Path("data/Mutagenicity")

    # Save/load preprocessed dataset to avoid recomputation
    import os
    processed_path = root / "mutagenicity_gnan_preprocessed.pt"
    if processed_path.exists():
        print(f"Loading preprocessed dataset from {processed_path}")
        try:
            dataset = torch.load(processed_path)
        except (pickle.UnpicklingError, RuntimeError):
            print("Could not load preprocessed file, re-creating...")
            os.remove(processed_path)
            dataset = TUDataset(root=str(root), name="Mutagenicity",
                                transform=PreprocessDistances())
            torch.save(dataset, processed_path)
    else:
        print("Preprocessing dataset and saving to disk...")
        dataset = TUDataset(root=str(root), name="Mutagenicity",
                            transform=PreprocessDistances())
        torch.save(dataset, processed_path)

    num_classes = dataset.num_classes
    in_channels = dataset.num_features

    print(f"Dataset info: {num_classes} classes, {in_channels} features")
    print(f"Dataset size: {len(dataset)} graphs")

    # Simple 80/10/10 split
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    n_train = int(0.8 * len(indices))
    n_val = int(0.1 * len(indices))

    # print("=" * 60)
    train_dataset = dataset[indices[:n_train]]
    val_dataset = dataset[indices[n_train:n_train + n_val]]
    test_dataset = dataset[indices[n_train + n_val:]]

    # standard PyTorch DataLoader with custom collate function
    train_loader = GNANDataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = GNANDataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = GNANDataLoader(test_dataset, batch_size=32, shuffle=False)

    model = TensorGNAN(
        in_channels=in_channels,
        out_channels=1 if num_classes == 2 else num_classes,
        n_layers=3,
        hidden_channels=64,
        dropout=0.3,
        normalize_rho=False,
        # Uncomment the following line to group features together:
        # feature_groups=[list(range(in_channels))] 
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4,
                                 weight_decay=5e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3)
    criterion = nn.BCEWithLogitsLoss()

    def evaluate(loader):
        model.eval()
        correct = 0
        for data in tqdm(loader, desc="Evaluating"):
            data = data.to(device)
            out = model(data)
            pred = out.squeeze() > 0
            correct += int((pred == data.y.to(device)).sum())
        return correct / len(loader.dataset)

    best_val_acc = 0.0
    for epoch in range(1, 11):
        model.train()
        total_loss = 0.0
        num_batches = 0

        for data in tqdm(train_loader, desc="Training"):
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out.squeeze(-1), data.y.to(device).float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        val_acc = evaluate(val_loader)
        test_acc = evaluate(test_loader)
        best_val_acc = max(best_val_acc, val_acc)

        print(f"Epoch {epoch:03d}  Loss {avg_loss:.4f}  "
              f"ValAcc {val_acc:.4f}  TestAcc {test_acc:.4f}")

        # Step the scheduler based on validation accuracy
        scheduler.step(val_acc)

    print("Best validation accuracy:", best_val_acc)
    print("Final test accuracy:", evaluate(test_loader))


if __name__ == "__main__":
    main()
