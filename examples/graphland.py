import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, average_precision_score, r2_score
from tqdm import tqdm

import torch_geometric.nn as pygnn
from torch_geometric.datasets import GraphLandDataset, graphland

GRAPHLAND_DATASETS = [
    'hm-categories',
    'pokec-regions',
    'web-topics',
    'tolokers-2',
    'city-reviews',
    'artnet-exp',
    'web-fraud',
    'hm-prices',
    'avazu-ctr',
    'city-roads-M',
    'city-roads-L',
    'twitch-views',
    'artnet-views',
    'web-traffic',
]


class Model(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int,
                 out_channels: int):
        super().__init__()
        self.conv = pygnn.GCNConv(in_channels, hidden_channels)
        self.head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels),
        )

    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        return self.head(self.conv(x, edge_index))


def _get_num_classes(dataset: GraphLandDataset) -> int:
    assert dataset.task != 'regression'
    targets = torch.cat([data.y for data in dataset], dim=0)
    return len(torch.unique(targets[~torch.isnan(targets)]))


def _get_model(dataset: GraphLandDataset) -> nn.Module:
    return Model(
        in_channels=dataset[0].x.shape[1],
        hidden_channels=256,
        out_channels=(_get_num_classes(dataset)
                      if dataset.task != 'regression' else 1),
    )


def _get_optimizer(model: nn.Module) -> optim.Optimizer:
    return optim.Adam(model.parameters(), lr=1e-3)


def _train_step(model: nn.Module, dataset: GraphLandDataset,
                optimizer: optim.Optimizer) -> float:
    def _compute_loss(outputs: torch.Tensor,
                      targets: torch.Tensor) -> torch.Tensor:
        if dataset.task == 'regression':
            return F.mse_loss(outputs, targets)
        else:
            return F.cross_entropy(outputs, targets.long())

    data = dataset[0]
    mask = data.train_mask if dataset.split != 'THI' else data.mask

    outputs = model(data.x, data.edge_index).squeeze()
    loss = _compute_loss(outputs[mask], data.y[mask])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.detach().cpu().item()


def _eval_step(model: nn.Module,
               dataset: GraphLandDataset) -> dict[str, float]:
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
    n_steps = 100
    dataset = GraphLandDataset(
        root='./datasets',
        split=split,
        name=name,
        to_undirected=True,
    )
    model = _get_model(dataset)
    model = model.cuda()
    dataset = dataset.copy().cuda()
    optimizer = _get_optimizer(model)

    best_metrics = {part: -float('inf') for part in ['train', 'val', 'test']}
    pbar = tqdm(range(n_steps))
    for _ in pbar:
        loss = _train_step(model, dataset, optimizer)
        curr_metrics = _eval_step(model, dataset)
        description = f'loss={loss:.4f}, ' + _format_metrics(curr_metrics)
        pbar.set_postfix_str(description)
        if curr_metrics['val'] > best_metrics['val']:
            best_metrics = curr_metrics

    print('Best metrics: ' + _format_metrics(best_metrics))
    return best_metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', choices=graphland.GRAPHLAND_DATASETS,
                        help='The name of dataset.', required=True)
    parser.add_argument('--split', choices=['RL', 'RH', 'TH', 'THI'],
                        help='The type of data split.', required=True)
    args = parser.parse_args()
    run_experiment(args.name, args.split)
