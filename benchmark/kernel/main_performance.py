import argparse
from itertools import product

import torch
from datasets import get_dataset
from gcn import GCN
from gin import GIN
from graph_sage import GraphSAGE
from train_eval import eval_acc, inference_run, train

import torch_geometric
from torch_geometric import seed_everything
from torch_geometric.loader import DataLoader
from torch_geometric.profile import rename_profile_file, timeit, torch_profile

seed_everything(0)

parser = argparse.ArgumentParser()
parser.add_argument(
    '--datasets', type=str, nargs='+',
    default=['MUTAG', 'PROTEINS', 'IMDB-BINARY', 'REDDIT-BINARY'])
parser.add_argument('--models', type=str, nargs='+',
                    default=['GCN', 'GraphSAGE', 'GIN'])
parser.add_argument('--layers', type=int, nargs='+', default=[1, 2, 3])
parser.add_argument('--hiddens', type=int, nargs='+', default=[16, 32])
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--warmup_profile', type=int, default=1,
                    help='Skip the first few runs')
parser.add_argument('--goal_accuracy', type=int, default=1,
                    help='The goal test accuracy')
parser.add_argument('--inference', action='store_true')
parser.add_argument('--profile', action='store_true')
parser.add_argument('--bf16', action='store_true')
parser.add_argument('--compile', action='store_true')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    amp = torch.cuda.amp.autocast(enabled=False)
else:
    amp = torch.cpu.amp.autocast(enabled=args.bf16)

MODELS = {
    'GCN': GCN,
    'GraphSAGE': GraphSAGE,
    'GIN': GIN,
}


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
    for dataset_name, model_name in product(args.datasets, args.models):
        dataset, train_loader, val_loader, test_loader = prepare_dataloader(
            dataset_name)
        Model = MODELS[model_name]

        for num_layers, hidden in product(args.layers, args.hiddens):
            print('--')
            print(f'{dataset_name} - {model_name}- {num_layers} - {hidden}')

            model = Model(dataset, num_layers, hidden).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            if args.compile:
                model = torch_geometric.compile(model)
            loss_list = []
            acc_list = []
            for epoch in range(1, args.epochs + 1):
                if epoch == args.epochs:
                    with timeit():
                        loss = train(model, optimizer, train_loader)
                else:
                    loss = train(model, optimizer, train_loader)

                with timeit(log=False) as t:
                    val_acc = eval_acc(model, val_loader)
                val_time = t.duration
                with timeit(log=False) as t:
                    test_acc = eval_acc(model, test_loader)
                test_time = t.duration

                if epoch >= args.warmup_profile:
                    loss_list.append(loss)
                    acc_list.append([val_acc, val_time, test_acc, test_time])

            if args.profile:
                with torch_profile():
                    train(model, optimizer, train_loader)
                rename_profile_file(model_name, dataset_name, str(num_layers),
                                    str(hidden), 'train')


@torch.no_grad()
def run_inference():
    for dataset_name, model_name in product(args.datasets, args.models):
        dataset, _, _, test_loader = prepare_dataloader(dataset_name)
        Model = MODELS[model_name]

        for num_layers, hidden in product(args.layers, args.hiddens):
            print('--')
            print(f'{dataset_name} - {model_name}- {num_layers} - {hidden}')

            model = Model(dataset, num_layers, hidden).to(device)
            if args.compile:
                model = torch_geometric.compile(model)
            with amp:
                for epoch in range(1, args.epochs + 1):
                    if epoch == args.epochs:
                        with timeit():
                            inference_run(model, test_loader, args.bf16)
                    else:
                        inference_run(model, test_loader, args.bf16)

                if args.profile:
                    with torch_profile():
                        inference_run(model, test_loader, args.bf16)
                    rename_profile_file(model_name, dataset_name,
                                        str(num_layers), str(hidden),
                                        'inference')


if not args.inference:
    run_train()
else:
    run_inference()
