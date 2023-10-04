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
from torch_geometric.nn import GCNConv, SoftMedianAggregation


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Cora')
parser.add_argument('--hidden_channels', type=int, default=64)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--temperature', type=float, default=0.5)
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

torch.random.manual_seed(args.seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Planetoid')
dataset = Planetoid(path, args.dataset, transform=T.NormalizeFeatures())
data = dataset[0].to(device)


transform = T.GDC(
    self_loop_weight=1,
    normalization_in='sym',
    normalization_out='col',
    diffusion_kwargs=dict(method='ppr', alpha=0.15, eps=1e-4),
    sparsification_kwargs=dict(method='topk', k=64, dim=0),
    exact=True,
)
data_gdc = transform(data).to(device)


# Required since weighted aggregations are not supported by default (yet)
class WeightedAggregationGCNConv(GCNConv):
    def propagate(self, edge_index: Tensor, x: Tensor, edge_weight: Tensor,
                  size=None, **kwargs) -> Tensor:
        size = self._check_input(edge_index, size)
        coll_dict = self._collect(self._user_args, edge_index, size,
                                  kwargs=dict(x=x, edge_weight=edge_weight))
        aggr_kwargs = self.inspector.distribute('aggregate', coll_dict)

        return self.aggr_module(coll_dict['x_j'], weight=edge_weight,
                                **aggr_kwargs)


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, normalize,
                 aggr=None):
        super().__init__()
        if aggr is not None:
            self.conv1 = WeightedAggregationGCNConv(
                in_channels, hidden_channels, normalize=normalize, aggr=aggr)
            self.conv2 = WeightedAggregationGCNConv(
                hidden_channels, out_channels, normalize=normalize, aggr=aggr)
        else:
            self.conv1 = GCNConv(in_channels, hidden_channels,
                                 normalize=normalize)
            self.conv2 = GCNConv(hidden_channels, out_channels,
                                 normalize=normalize)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight).relu()
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x


def train(model, data, epochs=200, lr=0.01, weight_decay=1e-3):
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val_acc = 0
    for _ in tqdm(range(epochs)):
        model.train()
        optimizer.zero_grad()
        pred = model(data.x, data.edge_index, data.edge_attr)
        loss = F.cross_entropy(pred[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        # For early stopping
        val_acc = test(model, data, mask='val_mask')
        if best_val_acc <= val_acc:
            best_val_acc = val_acc
            best_model_state_dict = {
                key: value.clone() for key, value in model.state_dict().items()}

    model.load_state_dict(best_model_state_dict)
    model.eval()


def accuracy(pred, y, mask):
    return (pred.argmax(-1)[mask] == y[mask]).float().mean()


@torch.no_grad()
def test(model, data, mask='test_mask'):
    model.eval()
    pred = model(data.x, data.edge_index, data.edge_attr)
    return float(accuracy(pred, data.y, data[mask]))


# The metric in PRBCD is assumed to be best if lower (like a loss).
def metric(*args, **kwargs):
    return -accuracy(*args, **kwargs)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    gcn = GCN(dataset.num_features, args.hidden_channels, dataset.num_classes,
              normalize=True).to(device)
    gdc = GCN(dataset.num_features, args.hidden_channels, dataset.num_classes,
              normalize=False).to(device)
    softmedian = SoftMedianAggregation(T=args.temperature)
    softmedian_gdc = GCN(dataset.num_features, args.hidden_channels,
                         dataset.num_classes, normalize=False,
                         aggr=softmedian).to(device)

    train(gcn, data, args.epochs, args.lr)
    train(gdc, data_gdc, args.epochs, args.lr)
    train(softmedian_gdc, data_gdc, args.epochs, args.lr)

    gcn_clean_acc = test(gcn, data)
    print(f'GCN clean accuracy: {gcn_clean_acc:.3f}')
    gdc_clean_acc = test(gdc, data_gdc)
    print(f'GDC clean accuracy: {gdc_clean_acc:.3f}')
    sm_gdc_clean_acc = test(softmedian_gdc, data_gdc)
    print(f'SoftMedian GDC clean accuracy: {sm_gdc_clean_acc:.3f}')

    # Perturb 10% of edges:
    global_budget = int(0.1 * data.edge_index.size(1) / 2)

    print('------------- GCN: Evasion -------------')
    prbcd = PRBCDAttack(gcn, block_size=1_000_000, metric=metric, lr=2_000)

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
    print('PRBCD: GCN accuracy dropped '
          f'from {gcn_clean_acc:.3f} to {pert_acc:.3f}')

    # A transfer attack is only performed for illustrative purposes
    print('------------ Transfer attack to GDC and SoftMedianGDC ------------')
    pert_data = transform(pert_data)

    pert_acc = test(gdc, pert_data)
    print('PRBCD: GDC accuracy dropped '
          f'from {gdc_clean_acc:.3f} to {pert_acc:.3f}')
    pert_acc = test(softmedian_gdc, pert_data)
    print('PRBCD: SoftMedian GDC accuracy dropped '
          f'from {sm_gdc_clean_acc:.3f} to {pert_acc:.3f}')
