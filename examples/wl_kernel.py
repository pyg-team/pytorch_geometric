import warnings
import argparse
import os.path as osp

import torch
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.exceptions import ConvergenceWarning
from torch_geometric.nn import WLConv
from torch_geometric.data import Batch
from torch_geometric.datasets import TUDataset

warnings.filterwarnings('ignore', category=ConvergenceWarning)

parser = argparse.ArgumentParser()
parser.add_argument('--runs', type=int, default=10)
args = parser.parse_args()

torch.manual_seed(42)

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'TU')
dataset = TUDataset(path, name='ENZYMES')
data = Batch.from_data_list(dataset)


class WL(torch.nn.Module):
    def __init__(self, num_layers):
        super().__init__()
        self.convs = torch.nn.ModuleList([WLConv() for _ in range(num_layers)])

    def forward(self, x, edge_index, batch=None):
        hists = []
        for conv in self.convs:
            x = conv(x, edge_index)
            hists.append(conv.histogram(x, batch, norm=True))
        return hists


wl = WL(num_layers=5)
hists = wl(data.x, data.edge_index, data.batch)

test_accs = torch.empty(args.runs, dtype=torch.float)

for run in range(1, args.runs + 1):
    perm = torch.randperm(data.num_graphs)
    val_index = perm[:data.num_graphs // 10]
    test_index = perm[data.num_graphs // 10:data.num_graphs // 5]
    train_index = perm[data.num_graphs // 5:]

    best_val_acc = 0

    for hist in hists:
        train_hist, train_y = hist[train_index], data.y[train_index]
        val_hist, val_y = hist[val_index], data.y[val_index]
        test_hist, test_y = hist[test_index], data.y[test_index]

        for C in [10**3, 10**2, 10**1, 10**0, 10**-1, 10**-2, 10**-3]:
            model = LinearSVC(C=C, tol=0.01, dual=True)
            model.fit(train_hist, train_y)
            val_acc = accuracy_score(val_y, model.predict(val_hist))
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                test_acc = accuracy_score(test_y, model.predict(test_hist))
                test_accs[run - 1] = test_acc

    print(f'Run: {run:02d}, Val: {best_val_acc:.4f}, Test: {test_acc:.4f}')

print(f'Final Test Performance: {test_accs.mean():.4f}±{test_accs.std():.4f}')
