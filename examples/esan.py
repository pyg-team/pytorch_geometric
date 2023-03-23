import argparse
import random

import numpy as np
import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import (
    DSnetwork,
    DSSnetwork,
    GCNConv,
    GINEConv,
    GraphConv,
)
from torch_geometric.transforms import subgraph_policy


class SimpleEvaluator():
    def __init__(self, task_type):
        self.task_type = task_type

    def acc(self, input_dict):
        y_true, y_pred = input_dict['y_true'], input_dict['y_pred']

        y_pred = (y_pred.mean(-1) > 0.5).astype(int)

        acc_list = []

        for i in range(y_true.shape[1]):
            is_labeled = y_true[:, i] == y_true[:, i]
            correct = y_true[is_labeled, i] == y_pred[is_labeled, i]
            acc_list.append(float(np.sum(correct)) / len(correct))

        return {'acc': sum(acc_list) / len(acc_list)}

    def mae(self, input_dict):
        y_true, y_pred = input_dict['y_true'], input_dict['y_pred']

        y_pred = np.mean(y_pred, axis=-1)

        return {'mae': np.average(np.abs(y_true - y_pred))}

    def eval(self, input_dict):
        if self.task_type == 'classification':
            return self.acc(input_dict)
        return self.mae(input_dict)


def separate_data(dataset, seed):
    indices = np.arange(len(dataset))
    labels = dataset.data.y.numpy()
    train_idx, test_idx, _, _ = train_test_split(indices, labels,
                                                 test_size=0.1,
                                                 random_state=seed,
                                                 shuffle=True, stratify=labels)

    return {'train': torch.tensor(train_idx), 'test': torch.tensor(test_idx)}


def get_data(args):

    # Subgraph generation policy
    if args.policy == 'original':
        pre_transform = None
    else:
        pre_transform = subgraph_policy(policy=args.policy,
                                        num_hops=args.num_hops)

    # Automatic dataloading and splitting
    dataset = TUDataset(
        root="dataset/" + args.policy,
        name=args.dataset,
        pre_transform=pre_transform,
    )

    # Ensure edge_attr is not considered
    dataset.data.edge_attr = None
    split_idx = separate_data(dataset, args.seed)

    train_loader = DataLoader(dataset[split_idx["train"]],
                              batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers,
                              follow_batch=['subgraph_id'])
    train_loader_eval = DataLoader(dataset[split_idx["train"]],
                                   batch_size=args.batch_size, shuffle=False,
                                   num_workers=args.num_workers,
                                   follow_batch=['subgraph_id'])
    test_loader = DataLoader(dataset[split_idx["test"]],
                             batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers,
                             follow_batch=['subgraph_id'])

    in_dim = dataset.num_features
    out_dim = 1 if args.dataset != "IMDB-MULTI" else 3
    task_type = 'classification'
    eval_metric = 'acc'

    return train_loader, train_loader_eval, test_loader, (in_dim, out_dim,
                                                          task_type,
                                                          eval_metric)


def get_model(args, in_dim, out_dim, device):
    if args.gnn_type == 'graphconv':
        GNNConv = GraphConv
    elif args.gnn_type == 'ginconv':
        GNNConv = GINEConv
    elif args.gnn_type == 'gcnconv':
        GNNConv = GCNConv
    else:
        raise ValueError('Undefined GNN type called {}'.format(args.gnn_type))

    if args.model == 'ds':
        model_name = DSnetwork
    elif args.model == 'dss':
        model_name = DSSnetwork
    else:
        raise ValueError('Undefined model type called {}'.format(args.model))

    model = model_name(num_layers=args.num_layer, in_dim=in_dim,
                       emb_dim=args.emb_dim, num_tasks=out_dim,
                       feature_encoder=lambda x: x, GNNConv=GNNConv).to(device)

    return model


def train(model, device, loader, optimizer, criterion, epoch):
    model.train()

    for batch in loader:
        batch = batch.to(device)
        pred = model(batch)
        optimizer.zero_grad()

        # ignore nan targets (unlabeled) when computing training loss.
        is_labeled = batch.y == batch.y

        y = batch.y.view(pred.shape).to(
            torch.float32) if pred.size(-1) == 1 else batch.y
        loss = criterion(pred.to(torch.float32)[is_labeled], y[is_labeled])

        loss.backward()
        optimizer.step()


@torch.no_grad()
def eval(model, device, loader, evaluator):
    model.eval()

    y_true = []
    y_pred = []

    for step, batch in enumerate(loader):
        batch = batch.to(device)
        pred = model(batch)

        y = batch.y.view(pred.shape) if pred.size(-1) == 1 else batch.y
        y_true.append(y.detach().cpu())
        y_pred.append(pred.detach().cpu())

    all_y_pred = torch.cat(y_pred, dim=0).unsqueeze(-1).numpy()

    y_true = torch.cat(y_true, dim=0).numpy()
    input_dict = {"y_true": y_true, "y_pred": all_y_pred}
    return evaluator.eval(input_dict)


def run(args, device):
    # set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    train_loader, train_loader_eval, test_loader, attributes = get_data(args)
    in_dim, out_dim, task_type, eval_metric = attributes

    evaluator = SimpleEvaluator(task_type)
    model = get_model(args, in_dim, out_dim, device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_step,
                                          gamma=args.decay_rate)

    if "classification" in task_type:
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        criterion = torch.nn.L1Loss()

    train_curve = []
    test_curve = []

    for epoch in tqdm(range(1, args.epochs + 1)):

        train(model, device, train_loader, optimizer, criterion, epoch=epoch)

        # Only valid_perf is used for TUD
        train_perf = eval(model, device, train_loader_eval, evaluator)
        test_perf = eval(model, device, test_loader, evaluator)

        scheduler.step()

        train_curve.append(train_perf[eval_metric])
        test_curve.append(test_perf[eval_metric])

    return (train_curve, test_curve)


def main():
    """ESAN (DS, DSS) Trainer example on TUDatasets (e.g. MUTAG).
    To reproduce model performance from paper run the following:
        python esan.py --dataset MUTAG --model dss --gnn_type graphconv
        python esan.py --dataset MUTAG --model ds --gnn_type graphconv
    """
    # Training settings
    parser = argparse.ArgumentParser(
        description='GNN baselines with Pytorch Geometrics')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--model', type=str, help='Type of model {ds, dss}')
    parser.add_argument(
        '--gnn_type', type=str,
        help='Type of convolution {graphconv, ginconv, gcnconv}')
    parser.add_argument('--drop_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument(
        '--num_layer', type=int, default=5,
        help='number of GNN message passing layers (default: 5)')
    parser.add_argument(
        '--emb_dim', type=int, default=64,
        help='dimensionality of hidden units in GNNs (default: 64)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='learning rate for training (default: 0.01)')
    parser.add_argument('--decay_rate', type=float, default=0.5,
                        help='decay rate for training (default: 0.5)')
    parser.add_argument('--decay_step', type=int, default=50,
                        help='decay step for training (default: 50)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')
    parser.add_argument('--dataset', type=str, default="MUTAG",
                        help='dataset name (default: MUTAG)')
    parser.add_argument(
        '--policy', type=str, default="edge_deletion",
        help='Subgraph selection policy in {edge_deletion, node_deletion, '
        'ego, ego_plus} (default: edge_deletion)')
    parser.add_argument(
        '--num_hops', type=int, default=2,
        help='Depth of the ego net if policy is ego (default: 2)')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed (default: 0)')

    args = parser.parse_args()

    device = torch.device(
        "cuda:" +
        str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    train_curve, test_curve = run(args, device)

    best_test_epoch = np.argmax(test_curve)
    best_train = max(train_curve)
    print(f'Best Loss = {best_train}, '
          f'Metric/train = {train_curve[best_test_epoch]}, '
          f'Metric/test = {test_curve[best_test_epoch]}')


if __name__ == "__main__":
    main()
