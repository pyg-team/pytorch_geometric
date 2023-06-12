import argparse

import torch
import torch.nn.functional as F
from citation import get_planetoid_dataset, random_planetoid_splits, run

from torch_geometric.nn import GATConv
from torch_geometric.profile import rename_profile_file

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--random_splits', action='store_true')
parser.add_argument('--runs', type=int, default=100)
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--early_stopping', type=int, default=100)
parser.add_argument('--hidden', type=int, default=8)
parser.add_argument('--dropout', type=float, default=0.6)
parser.add_argument('--no_normalize_features', action='store_true')
parser.add_argument('--heads', type=int, default=8)
parser.add_argument('--output_heads', type=int, default=1)
parser.add_argument('--inference', action='store_true')
parser.add_argument('--profile', action='store_true')
parser.add_argument('--bf16', action='store_true')
parser.add_argument('--compile', action='store_true')
args = parser.parse_args()


class Net(torch.nn.Module):
    def __init__(self, dataset):
        super().__init__()
        self.conv1 = GATConv(dataset.num_features, args.hidden,
                             heads=args.heads, dropout=args.dropout)
        self.conv2 = GATConv(args.hidden * args.heads, dataset.num_classes,
                             heads=args.output_heads, concat=False,
                             dropout=args.dropout)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=args.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=args.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


dataset = get_planetoid_dataset(args.dataset, not args.no_normalize_features)
permute_masks = random_planetoid_splits if args.random_splits else None
run(dataset, Net(dataset), args.runs, args.epochs, args.lr, args.weight_decay,
    args.early_stopping, args.inference, args.profile, args.bf16, args.compile,
    permute_masks)

if args.profile:
    rename_profile_file('citation', GATConv.__name__, args.dataset,
                        str(args.random_splits),
                        'inference' if args.inference else 'train')
