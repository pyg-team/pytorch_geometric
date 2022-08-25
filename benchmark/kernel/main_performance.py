import argparse
from itertools import product

import torch
from datasets import get_dataset
from gcn import GCN
from gin import GIN
from graph_sage import GraphSAGE
from train_eval import eval_acc, inference_run, train

from torch_geometric import seed_everything
from torch_geometric.loader import DataLoader
from torch_geometric.profile import (
    get_stats_summary,
    profileit,
    rename_profile_file,
    timeit,
    torch_profile,
)

seed_everything(0)

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--warmup_profile', type=int, default=1,
                    help='Skip the first few runs')
parser.add_argument('--goal_accuracy', type=int, default=1,
                    help='The goal test accuracy')
parser.add_argument('--inference', action='store_true')
parser.add_argument('--profile', action='store_true')
args = parser.parse_args()

layers = [1, 2, 3]
hiddens = [16, 32]

datasets = ['MUTAG', 'PROTEINS', 'IMDB-BINARY', 'REDDIT-BINARY']

nets = [
    GCN,
    GraphSAGE,
    GIN,
]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def prepare_dataloader(dataset_name):
    dataset = get_dataset(dataset_name, sparse=True)
    num_train = int(len(dataset) * 0.8)
    num_val = int(len(dataset) * 0.1)

    train_dataset = dataset[:num_train]
    val_dataset = dataset[num_train:num_train + num_val]
    test_dataset = dataset[num_train + num_val:]

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False)
    return dataset, train_loader, val_loader, test_loader


def run_train():
    for dataset_name, Net in product(datasets, nets):
        dataset, train_loader, val_loader, test_loader = prepare_dataloader(
            dataset_name)

        for num_layers, hidden in product(layers, hiddens):
            print(
                f'--\n{dataset_name} - {Net.__name__}- {num_layers} - {hidden}'
            )

            model = Net(dataset, num_layers, hidden).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

            stats_list = []
            acc_list = []
            for epoch in range(1, args.epochs + 1):
                loss, stats = train(model, optimizer, train_loader)
                with timeit(log=False) as t:
                    val_acc = eval_acc(model, val_loader)
                val_time = t.duration
                with timeit(log=False) as t:
                    test_acc = eval_acc(model, test_loader)
                test_time = t.duration

                if epoch >= args.warmup_profile:
                    stats_list.append(stats)
                    acc_list.append([val_acc, val_time, test_acc, test_time])

            stats_summary = get_stats_summary(stats_list)
            print(stats_summary)


@torch.no_grad()
def run_inference():
    for dataset_name, Net in product(datasets, nets):
        dataset, _, _, test_loader = prepare_dataloader(dataset_name)

        for num_layers, hidden in product(layers, hiddens):
            print(
                f'--\n{dataset_name} - {Net.__name__}- {num_layers} - {hidden}'
            )

            model = Net(dataset, num_layers, hidden).to(device)

            for epoch in range(1, args.epochs + 1):
                if epoch == args.epochs:
                    with timeit():
                        inference_run(model, test_loader)
                else:
                    inference_run(model, test_loader)

            if args.profile:
                with torch_profile():
                    inference_run(model, test_loader)
                rename_profile_file(Net.__name__, dataset_name,
                                    str(num_layers), str(hidden))


if not args.inference:
    # Decorate train functions:
    train = profileit()(train)
    run_train()
else:
    run_inference()
