import argparse

import torch
import torch.nn.functional as F
from citation import get_planetoid_dataset, random_planetoid_splits, run

from torch_geometric.nn import SGConv

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--random_splits', type=bool, default=False)
parser.add_argument('--runs', type=int, default=100)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--early_stopping', type=int, default=10)
parser.add_argument('--normalize_features', type=bool, default=False)
parser.add_argument('--K', type=int, default=2)
parser.add_argument('--inference', type=bool, default=False)
parser.add_argument('--profile', type=bool, default=False) # Currently support profile in inference
args = parser.parse_args()


class Net(torch.nn.Module):
    def __init__(self, dataset):
        super().__init__()
        self.conv1 = SGConv(dataset.num_features, dataset.num_classes,
                            K=args.K, cached=True)

    def reset_parameters(self):
        self.conv1.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        return F.log_softmax(x, dim=1)


dataset = get_planetoid_dataset(args.dataset, args.normalize_features)
permute_masks = random_planetoid_splits if args.random_splits else None
run(dataset, Net(dataset), args.runs, args.epochs, args.lr, args.weight_decay,
    args.early_stopping, args.inference, args.profile, permute_masks)

if args.profile:
    import os
    import pathlib
    profile_dir = str(pathlib.Path.cwd()) + '/'
    profile_file = profile_dir + 'profile-citation-SGC-' + args.dataset + '-random_splits-' + str(args.random_splits) + '.log'
    timeline_file = profile_dir + 'profile-citation-SGC-' + args.dataset + '-random_splits-' + str(args.random_splits) + '.json'
    os.rename('profile.log', profile_file)
    os.rename('timeline.json', timeline_file)
