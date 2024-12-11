# Example usage of PPRGo (from the paper "Scaling Graph Neural Networks
# with Approximate PageRank). This trains on a small subset of the Reddit
# graph dataset and predicts on the rest.

import argparse
import os.path as osp
import time

import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

from torch_geometric.data import Data
from torch_geometric.datasets import Reddit
from torch_geometric.nn.models import PPRGo, pprgo_prune_features
from torch_geometric.transforms import GDC
from torch_geometric.utils import subgraph

# Parameters
parser = argparse.ArgumentParser()
parser.add_argument('--alpha', type=float, default=0.5)
parser.add_argument('--eps', type=float, default=1e-3)
parser.add_argument('--topk', type=int, default=32)
parser.add_argument('--n_train', type=int, default=1000)
parser.add_argument('--hidden_size', type=int, default=64)
parser.add_argument('--n_layers', type=int, default=2)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--n_epochs', type=int, default=50)
parser.add_argument('--n_power_iters', type=int, default=2)
parser.add_argument('--frac_predict', type=float, default=0.2)
args = parser.parse_args()

# Load Reddit dataset
s = time.time()
name = 'Reddit'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', name)
data = Reddit(path)[0]
print(f'Loaded Reddit dataset in {round(time.time() - s, 2)}s')

# Get random split of train/test nodes (|train| << |test|)
s = time.time()
ind = torch.randperm(len(data.x), dtype=torch.long)
train_idx = ind[:args.n_train]
test_idx = ind[args.n_train:]

train_edge_index, _ = subgraph(train_idx, data.edge_index, relabel_nodes=True)
train = Data(x=data.x[train_idx], edge_index=train_edge_index,
             y=data.y[train_idx])
print(f'Split data into {len(train_idx)} train and {len(test_idx)} test nodes '
      f'in {round(time.time() - s, 2)}s')

# Set up ppr transform via gdc
s = time.time()
ppr = GDC(
    exact=False, normalization_in='sym', normalization_out=None,
    diffusion_kwargs=dict(
        method='ppr',
        alpha=args.alpha,
        eps=args.eps,
    ), sparsification_kwargs=dict(method='topk', topk=args.topk))
train = ppr(train)
print(f'Ran PPR on {args.n_train} train nodes in {round(time.time() - s, 2)}s')

# Set up model and optimizer
model = PPRGo(num_features=data.x.shape[1], num_classes=data.y.max() + 1,
              hidden_size=args.hidden_size, n_layers=args.n_layers,
              dropout=args.dropout)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-3)

# Run training, single-batch full graph
s = time.time()
print()
print('=' * 40)
model.train()
train = pprgo_prune_features(train)
for i in range(1, args.n_epochs + 1):
    optimizer.zero_grad()
    pred = model(train.x, train.edge_index, train.edge_attr)

    loss = F.cross_entropy(pred, train.y)
    top1 = torch.argmax(pred, dim=1)
    acc = (torch.sum(top1 == train.y) / len(train.y)).item()

    loss.backward()
    optimizer.step()
    if i % 10 == 0:
        print(f'Epoch {i}: loss = {round(loss.item(), 3)}, '
              f'acc = {round(acc, 3)}')

print('=' * 40)
print(f'Finished training in {round(time.time() - s, 2)}s')

# Do inference on the test graph and report performance
s = time.time()
preds = model.predict_power_iter(data.x, data.edge_index, args.n_power_iters,
                                 args.alpha, args.frac_predict)
preds = preds.argmax(1)
labels = data.y

acc_train = 100 * accuracy_score(labels[train_idx], preds[train_idx])
acc_test = 100 * accuracy_score(labels[test_idx], preds[test_idx])

f1_train = f1_score(labels[train_idx], preds[train_idx], average='macro')
f1_test = f1_score(labels[test_idx], preds[test_idx], average='macro')

print()
print('=' * 40)
print(f'Accuracy ... train: {acc_train:.1f}%, test: {acc_test:.1f}%')
print(f'F1 score ... train: {f1_train:.3f}, test: {f1_test:.3f}')
print('=' * 40)
print(f'Ran inference in {round(time.time() - s, 2)}s')
