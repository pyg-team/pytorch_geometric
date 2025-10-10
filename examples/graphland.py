import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, average_precision_score, r2_score
from tqdm import tqdm

from torch_geometric.datasets import GraphLandDataset
from torch_geometric.nn import GCNConv


class Model(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        self.conv = GCNConv(in_channels, hidden_channels)
        self.head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        return self.head(self.conv(x, edge_index))


def _get_num_classes(dataset: GraphLandDataset) -> int:
    assert dataset.task != 'regression'
    targets = torch.cat([data.y for data in dataset], dim=0)
    return len(torch.unique(targets[~torch.isnan(targets)]))


def _train_step(
    model: nn.Module,
    dataset: GraphLandDataset,
    optimizer: optim.Optimizer,
) -> torch.Tensor:
    data = dataset[0]
    mask = data.train_mask if dataset.split != 'THI' else data.mask
    optimizer.zero_grad()
    outputs = model(data.x, data.edge_index).squeeze()

    if dataset.task == 'regression':
        loss = F.mse_loss(outputs[mask], data.y[mask])
    else:
        loss = F.cross_entropy(outputs[mask], data.y[mask].long())

    loss.backward()
    optimizer.step()
    return loss


def _eval_step(
    model: nn.Module,
    dataset: GraphLandDataset,
) -> dict[str, float]:
    def _compute_metric(outputs: np.ndarray, targets: np.ndarray) -> float:
        if dataset.task == 'regression':
            return float(r2_score(targets, outputs))

        elif dataset.task == 'binary_classification':
            predictions = outputs[:, 1]
            return float(average_precision_score(targets, predictions))

        else:
            predictions = np.argmax(outputs, axis=1)
            return float(accuracy_score(targets, predictions))

    metrics = dict()
    for idx, part in enumerate(['train', 'val', 'test']):
        if dataset.split == 'THI':
            data = dataset[idx]
            mask = data.mask
        else:
            data = dataset[0]
            mask = getattr(data, f'{part}_mask')

        outputs = model(data.x, data.edge_index).squeeze()
        metrics[part] = _compute_metric(
            outputs[mask].detach().cpu().numpy(),
            data.y[mask].cpu().numpy(),
        )
    return metrics


def _format_metrics(metrics: dict[str, float]) -> str:
    return ', '.join(f'{part}={metrics[part] * 100.0:.2f}'
                     for part in ['train', 'val', 'test'])


def run_experiment(name: str, split: str) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_steps = 100
    dataset = GraphLandDataset(
        root='./data',
        split=split,
        name=name,
        to_undirected=True,
    )
    model = Model(
        in_channels=dataset[0].x.shape[1],
        hidden_channels=256,
        out_channels=(_get_num_classes(dataset)
                      if dataset.task != 'regression' else 1),
    ).to(device)
    dataset = dataset.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    best_metrics = {part: -float('inf') for part in ['train', 'val', 'test']}
    pbar = tqdm(range(n_steps))
    for _ in pbar:
        loss = _train_step(model, dataset, optimizer)
        curr_metrics = _eval_step(model, dataset)
        pbar.set_postfix_str(f'loss={loss.detach().cpu().item():.4f}, ' +
                             _format_metrics(curr_metrics))
        if curr_metrics['val'] > best_metrics['val']:
            best_metrics = curr_metrics

    print('Best metrics: ' + _format_metrics(best_metrics))
    return best_metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--name',
        choices=list(GraphLandDataset.GRAPHLAND_DATASETS.keys()),
        help='The name of dataset.',
        required=True,
    )
    parser.add_argument(
        '--split',
        choices=['RL', 'RH', 'TH', 'THI'],
        help='The type of data split.',
        required=True,
    )
    args = parser.parse_args()
    run_experiment(args.name, args.split)
