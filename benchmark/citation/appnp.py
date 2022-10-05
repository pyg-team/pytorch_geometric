import argparse

import torch
import torch.nn.functional as F
from citation import get_planetoid_dataset, random_planetoid_splits, run
from torch.nn import Linear

from torch_geometric.nn import APPNP
from torch_geometric.profile import rename_profile_file

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--random_splits', action='store_true')
parser.add_argument('--runs', type=int, default=100)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--early_stopping', type=int, default=10)
parser.add_argument('--hidden', type=int, default=64)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--no_normalize_features', action='store_true')
parser.add_argument('--K', type=int, default=10)
parser.add_argument('--alpha', type=float, default=0.1)
parser.add_argument('--inference', action='store_true')
parser.add_argument('--profile', action='store_true')
parser.add_argument('--bf16', action='store_true')
args = parser.parse_args()


class Net(torch.nn.Module):
    def __init__(self, dataset):
        super().__init__()
        self.lin1 = Linear(dataset.num_features, args.hidden)
        self.lin2 = Linear(args.hidden, dataset.num_classes)
        self.prop1 = APPNP(args.K, args.alpha)

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=args.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=args.dropout, training=self.training)
        x = self.lin2(x)
        x = self.prop1(x, edge_index)
        return F.log_softmax(x, dim=1)


dataset = get_planetoid_dataset(args.dataset, not args.no_normalize_features)
permute_masks = random_planetoid_splits if args.random_splits else None
run(dataset, Net(dataset), args.runs, args.epochs, args.lr, args.weight_decay,
    args.early_stopping, args.inference, args.profile, args.bf16,
    permute_masks)

if args.profile:
    rename_profile_file('citation', APPNP.__name__, args.dataset,
                        str(args.random_splits))
