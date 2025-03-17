"""This is an example of using mixup for
node classification task.
"""

import argparse
import copy
import os.path as osp

import numpy as np
import torch
import torch.nn.functional as F

from torch_geometric.contrib.nn import NodeMixup
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="NodeMixup Training")

    # Dataset parameters
    parser.add_argument(
        '--dataset',
        type=str,
        default='Pubmed',
        choices=['Cora', 'CiteSeer', 'Pubmed'],
        help='Name of the dataset to use (default: Pubmed)',
    )

    # Training hyperparameters
    parser.add_argument('--num_layers', type=int, default=3,
                        help='Number of layers (default: 3)')
    parser.add_argument('--hidden_channels', type=int, default=256,
                        help='Number of hidden channels (default: 256)')
    parser.add_argument('--dropout', type=float, default=0.4,
                        help='Dropout rate (default: 0.4)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--epochs', type=int, default=300,
                        help='Number of training epochs (default: 300)')

    # Mixup parameters
    parser.add_argument(
        '--alpha',
        type=float,
        default=4.0,
        help='Alpha parameter for Beta distribution in mixup (default: 4.0)',
    )

    # Split ratios
    parser.add_argument('--train_ratio', type=float, default=0.6,
                        help='Ratio of training data (default: 0.6)')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                        help='Ratio of validation data (default: 0.2)')

    # Miscellaneous
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        choices=['cuda', 'cpu'],
        help='Device to use for computation (default: cuda if available)',
    )

    return parser.parse_args()


def shuffle_data(data: Data, seed: int = None):
    """Get the shuffled version of the input graph."""
    if seed is not None:
        np.random.seed(seed)
    data = copy.deepcopy(data)
    id_new_value_old = np.arange(data.num_nodes)
    train_id_shuffle = copy.deepcopy(data.train_id)
    np.random.shuffle(train_id_shuffle)
    id_new_value_old[data.train_id] = train_id_shuffle

    row, col = data.edge_index[0].cpu(), data.edge_index[1].cpu()
    id_old_value_new = torch.zeros(id_new_value_old.shape[0], dtype=torch.long)
    id_old_value_new[id_new_value_old] = torch.arange(
        id_new_value_old.shape[0], dtype=torch.long)
    row, col = id_old_value_new[row], id_old_value_new[col]
    data.edge_index = torch.stack([row, col], dim=0)

    return data, id_new_value_old


def main():
    args = parse_args()

    # Set random seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Load dataset
    dataset_name = args.dataset
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data',
                    dataset_name)
    dataset = Planetoid(path, dataset_name, transform=NormalizeFeatures())
    data = dataset[0]

    # Split dataset
    node_id = np.arange(data.num_nodes)
    np.random.shuffle(node_id)
    train_end = int(data.num_nodes * args.train_ratio)
    val_end = train_end + int(data.num_nodes * args.val_ratio)
    data.train_id = node_id[:train_end]
    data.val_id = node_id[train_end:val_end]
    data.test_id = node_id[val_end:]

    # Initialize the model
    model = NodeMixup(
        num_layers=args.num_layers,
        in_channels=dataset.num_node_features,
        hidden_channels=args.hidden_channels,
        out_channels=dataset.num_classes,
        dropout=args.dropout,
    )

    # Define optimizer and device
    device = torch.device(args.device)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    def train(data):
        """Training function."""
        model.train()
        lam = np.random.beta(args.alpha, args.alpha)

        data_b, id_new_value_old = shuffle_data(data, seed=args.seed)
        data = data.to(device)
        data_b = data_b.to(device)

        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data_b.edge_index, lam,
                    id_new_value_old)
        loss = (F.nll_loss(out[data.train_id], data.y[data.train_id]) * lam +
                F.nll_loss(out[data.train_id], data_b.y[data.train_id]) *
                (1 - lam))

        loss.backward()
        optimizer.step()

        return loss.item()

    @torch.no_grad()
    def test(data):
        """Testing function."""
        model.eval()
        out = model(data.x, data.edge_index, data.edge_index, 1,
                    np.arange(data.num_nodes))
        pred = out.argmax(dim=-1)
        correct = pred.eq(data.y)
        accs = []
        for id_ in [data.train_id, data.val_id, data.test_id]:
            acc = correct[id_].sum().item() / id_.shape[0]
            accs.append(acc)
        return accs

    # Training loop
    best_val_acc = 0
    best_test_acc = 0
    for epoch in range(1, args.epochs + 1):
        loss = train(data)
        train_acc, val_acc, test_acc = test(data)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}, "
                  f"Train Acc: {train_acc:.4f}"
                  f"Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}")

    print(f"Best Val Acc: {best_val_acc:.4f}, Corresponding Test Acc: "
          f"{best_test_acc:.4f}")


if __name__ == '__main__':
    main()
