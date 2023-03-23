import argparse
import copy
import os.path as osp

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Adam
from tqdm import tqdm

import torch_geometric.transforms as T
from torch_geometric.contrib.nn import PRBCDAttack
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.nn.aggr import (
    SoftMedianAggregation,
    WeightedMeanAggregation,
)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Cora')
parser.add_argument('--hidden_channels', type=int, default=16)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--epochs', type=int, default=200)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Planetoid')
dataset = Planetoid(path, args.dataset, transform=T.NormalizeFeatures())
data = dataset[0]

transform = T.GDC(
    self_loop_weight=1,
    normalization_in='sym',
    normalization_out='col',
    diffusion_kwargs=dict(method='ppr', alpha=0.1, eps=1e-4),
    sparsification_kwargs=dict(method='topk', k=64, dim=0),
    exact=False,
)
data_gdc = transform(data).to(device)

aggr = SoftMedianAggregation(T=1)


class CustomGCNConv(GCNConv):
    def message(self, x_j: Tensor) -> Tensor:
        # Do no multiply with weight s.t. we can use weighted aggregations
        return x_j


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, normalize,
                 aggr):
        super().__init__()
        self.conv1 = CustomGCNConv(in_channels, hidden_channels,
                                   normalize=normalize, aggr=aggr)
        self.conv2 = CustomGCNConv(hidden_channels, out_channels,
                                   normalize=normalize, aggr=aggr)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight).relu()
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x


def train(model, data, epochs=200, lr=0.01, weight_decay=5e-4):
    model.train()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in tqdm(range(epochs)):
        optimizer.zero_grad()
        pred = model(data.x, data.edge_index, data.edge_attr)
        loss = F.cross_entropy(pred[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
    model.eval()


def accuracy(pred, y, mask):
    return (pred.argmax(-1)[mask] == y[mask]).float().mean()


@torch.no_grad()
def test(model, data):
    model.eval()
    pred = model(data.x, data.edge_index, data.edge_attr)
    return float(accuracy(pred, data.y, data.test_mask))


# The metric in PRBCD is assumed to be best if lower (like a loss).
def metric(*args, **kwargs):
    return -accuracy(*args, **kwargs)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    gcn = GCN(dataset.num_features, args.hidden_channels, dataset.num_classes,
              normalize=True, aggr=WeightedMeanAggregation()).to(device)
    softmedian_gdc = GCN(dataset.num_features, args.hidden_channels,
                         dataset.num_classes, normalize=False,
                         aggr=SoftMedianAggregation(T=1)).to(device)

    train(gcn, data, args.epochs, args.lr)

    train(softmedian_gdc, data_gdc, args.epochs, args.lr)

    # Perturb 5% of edges:
    global_budget = int(0.05 * data.edge_index.size(1) / 2)

    print('------------- GCN: Evasion -------------')
    prbcd = PRBCDAttack(gcn, block_size=250_000, metric=metric, lr=2_000)

    clean_acc = test(gcn, data)
    print(f'Clean accuracy: {clean_acc:.3f}')

    # PRBCD: Attack test set:
    pert_edge_index, perts = prbcd.attack(
        data.x,
        data.edge_index,
        data.y,
        budget=global_budget,
        idx_attack=data.test_mask,
    )

    pert_data = copy.copy(data)
    pert_data.edge_index = pert_edge_index
    pert_data.edge_attr = None
    pert_acc = test(gcn, pert_data)
    print(f'PRBCD: Accuracy dropped from {clean_acc:.3f} to {pert_acc:.3f}')

    print('------------- Transfer attack to SoftMedianGDC -------------')

    clean_acc = test(softmedian_gdc, data)
    print(f'Clean accuracy: {clean_acc:.3f}')

    pert_data = transform(pert_data)
    pert_acc = test(softmedian_gdc, pert_data)
    print(f'PRBCD: Accuracy dropped from {clean_acc:.3f} to {pert_acc:.3f}')
