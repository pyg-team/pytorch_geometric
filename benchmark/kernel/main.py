from itertools import product

import argparse
from datasets import get_dataset
from train_eval import cross_validation_with_val_set

from gcn import GCN, GCNWithJK
from graph_sage import GraphSAGE, GraphSAGEWithJK
from gin import GIN0, GIN0WithJK, GIN, GINWithJK
from graclus import Graclus
from top_k import TopK
from sag_pool import SAGPool
from diff_pool import DiffPool
from edge_pool import EdgePool
from global_attention import GlobalAttentionNet
from set2set import Set2SetNet
from sort_pool import SortPool
from asap import ASAP

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--lr_decay_factor', type=float, default=0.5)
parser.add_argument('--lr_decay_step_size', type=int, default=50)
args = parser.parse_args()

layers = [1, 2, 3, 4, 5]
hiddens = [16, 32, 64, 128]
datasets = ['MUTAG', 'PROTEINS', 'IMDB-BINARY', 'REDDIT-BINARY']  # , 'COLLAB']
nets = [
    GCNWithJK,
    GraphSAGEWithJK,
    GIN0WithJK,
    GINWithJK,
    Graclus,
    TopK,
    SAGPool,
    DiffPool,
    EdgePool,
    GCN,
    GraphSAGE,
    GIN0,
    GIN,
    GlobalAttentionNet,
    Set2SetNet,
    SortPool,
    ASAP,
]


def logger(info):
    fold, epoch = info['fold'] + 1, info['epoch']
    val_loss, test_acc = info['val_loss'], info['test_acc']
    print(f'{fold:02d}/{epoch:03d}: Val Loss: {val_loss:.4f}, '
          f'Test Accuracy: {test_acc:.3f}')


results = []
for dataset_name, Net in product(datasets, nets):
    best_result = (float('inf'), 0, 0)  # (loss, acc, std)
    print(f'--\n{dataset_name} - {Net.__name__}')
    for num_layers, hidden in product(layers, hiddens):
        dataset = get_dataset(dataset_name, sparse=Net != DiffPool)
        model = Net(dataset, num_layers, hidden)
        loss, acc, std = cross_validation_with_val_set(
            dataset,
            model,
            folds=10,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            lr_decay_factor=args.lr_decay_factor,
            lr_decay_step_size=args.lr_decay_step_size,
            weight_decay=0,
            logger=None,
        )
        if loss < best_result[0]:
            best_result = (loss, acc, std)

    desc = f'{best_result[1]:.3f} ± {best_result[2]:.3f}'
    print(f'Best result - {desc}')
    results += [f'{dataset_name} - {model}: {desc}']
results = '\n'.join(results)
print(f'--\n{results}')
