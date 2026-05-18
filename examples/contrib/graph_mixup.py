"""This is an example of using mixup for
graph classification task.
"""

import argparse
import os.path as osp

import numpy as np
import torch
import torch.nn.functional as F

from torch_geometric.contrib.nn import GraphMixup
from torch_geometric.data import Batch, DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.nn import GATConv, GCNConv, SAGEConv


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="GraphMixup Training Example")
    parser.add_argument(
        '--dataset',
        type=str,
        default='PROTEINS',
        choices=['PROTEINS', 'DD'],
        help='Dataset name for graph classification (default: PROTEINS)',
    )
    parser.add_argument(
        '--gnn',
        type=str,
        default='GCN',
        choices=['GCN', 'GAT', 'GraphSAGE'],
        help='GNN backbone to use (default: GCN)',
    )
    parser.add_argument(
        '--num_layers',
        type=int,
        default=3,
        help='Number of layers (default: 3)',
    )
    parser.add_argument(
        '--hidden_channels',
        type=int,
        default=64,
        help='Hidden channels (default: 64)',
    )
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.5,
        help='Dropout rate (default: 0.5)',
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        help='Learning rate (default: 0.001)',
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs (default: 50)',
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=2.0,
        help='Alpha parameter for Beta distribution in mixup (default: 2.0)',
    )
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        choices=['cuda', 'cpu'],
        help='Device (default: cuda if available)',
    )
    return parser.parse_args()


def get_gnn_layer(gnn_type: str):
    """Retrieve the appropriate GNN layer based on the gnn_type."""
    if gnn_type == 'GCN':
        return GCNConv
    elif gnn_type == 'GAT':
        return GATConv
    elif gnn_type == 'GraphSAGE':
        return SAGEConv
    else:
        raise ValueError(f"Unsupported GNN type: {gnn_type}")


def shuffle_batch(data_list, seed=None):
    """Shuffle the graph batches."""
    if seed is not None:
        np.random.seed(seed)
    indices = np.arange(len(data_list))
    np.random.shuffle(indices)
    shuffled_list = [data_list[i] for i in indices]
    return shuffled_list, indices


def main():
    args = parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load dataset
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data',
                    args.dataset)
    dataset = TUDataset(path, name=args.dataset).shuffle()

    # Split dataset into train/val/test
    train_size = int(len(dataset) * 0.6)
    val_size = int(len(dataset) * 0.2)

    train_dataset = dataset[:train_size]
    val_dataset = dataset[train_size:train_size + val_size]
    test_dataset = dataset[train_size + val_size:]

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    device = torch.device(args.device)

    gnn_layer = get_gnn_layer(args.gnn)
    model = GraphMixup(
        num_layers=args.num_layers,
        in_channels=dataset.num_node_features,
        hidden_channels=args.hidden_channels,
        out_channels=dataset.num_classes,
        conv_layer=gnn_layer,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    def train():
        """Train the model."""
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            # Shuffle batch to create a second batch
            data_list = batch.to_data_list()
            data_b_list, _ = shuffle_batch(data_list, seed=args.seed)
            batch_b = Batch.from_data_list(data_b_list).to(device)

            lam = np.random.beta(args.alpha, args.alpha)

            out = model(
                batch.x,
                batch.edge_index,
                batch.batch,
                batch_b.x,
                batch_b.edge_index,
                batch_b.batch,
                lam,
            )

            # Mixup loss
            y = batch.y
            y_b = batch_b.y
            loss = lam * F.nll_loss(out, y) + (1 - lam) * F.nll_loss(out, y_b)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch.num_graphs

        return total_loss / len(train_loader.dataset)

    @torch.no_grad()
    def eval(loader):
        """Evaluate the model."""
        model.eval()
        correct = 0
        total = 0
        for batch in loader:
            batch = batch.to(device)
            out = model(
                batch.x,
                batch.edge_index,
                batch.batch,
                batch.x,
                batch.edge_index,
                batch.batch,
                lam=1.0,
            )
            pred = out.argmax(dim=-1)
            correct += pred.eq(batch.y).sum().item()
            total += batch.num_graphs
        return correct / total

    best_val_acc = 0
    best_test_acc = 0
    for epoch in range(1, args.epochs + 1):
        loss = train()
        val_acc = eval(val_loader)
        test_acc = eval(test_loader)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}, "
                  f"Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}")

    print(f"Best Val Acc: {best_val_acc:.4f}, "
          f"Corresponding Test Acc: {best_test_acc:.4f}")


if __name__ == "__main__":
    main()
