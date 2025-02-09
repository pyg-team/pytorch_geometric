import argparse

import torch
import torch.nn.functional as F
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset

from torch_geometric import seed_everything
from torch_geometric.nn.models import Polynormer as Net
from torch_geometric.utils import (
    add_self_loops,
    remove_self_loops,
    to_undirected,
)


def main(args):
    # load dataset ============================================================
    dataset = PygNodePropPredDataset(name=args.dataset, root=args.data_dir)
    g = dataset[0]
    # get the splits
    split_idx = dataset.get_idx_split()
    # basic information of datasets
    n = g.num_nodes
    e = g.num_edges
    # infer the number of classes for non one-hot and one-hot labels
    c = g.y.max().item() + 1
    d = g.x.size(1)

    print(f'dataset {args.dataset} | #nodes {n} | #edges {e} '
          f'| #node feats {d} | #classes {c}')

    g.edge_index = to_undirected(g.edge_index)
    g.edge_index, _ = remove_self_loops(g.edge_index)
    g.edge_index, _ = add_self_loops(g.edge_index, num_nodes=n)

    # load device ==========================================================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    g = g.to(device)

    # define model ===================================================
    model = Net(
        d,
        args.hidden_channels,
        c,
        local_layers=args.local_layers,
        global_layers=args.global_layers,
        in_dropout=args.in_dropout,
        dropout=args.dropout,
        global_dropout=args.global_dropout,
        heads=args.num_heads,
        beta=args.beta,
        pre_ln=args.pre_ln,
        post_bn=args.post_bn,
        local_attn=args.local_attn,
    ).to(device)
    criterion = torch.torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        weight_decay=args.weight_decay,
        lr=args.lr,
    )

    # define evaluator ===============================================
    evaluator = Evaluator(name=args.dataset)

    def evaluate(out, split):
        assert split in ['train', 'valid', 'test']
        y_true = g.y.cpu()
        y_pred = out.argmax(dim=-1, keepdim=True)
        return evaluator.eval({
            'y_true': y_true[split_idx[split]],
            'y_pred': y_pred[split_idx[split]],
        })['acc']

    # training loop ==================================================
    train_idx = split_idx['train'].to(device)
    best_val, best_test = 0, 0
    for epoch in range(args.local_epochs + args.global_epochs):
        if epoch == args.local_epochs:
            print('start global attention!')
            model._global = True
        model.train()
        optimizer.zero_grad()
        out = model(g.x, g.edge_index)
        out = F.log_softmax(out, dim=-1)
        loss = criterion(out[train_idx], g.y[train_idx].view(-1))
        loss.backward()
        optimizer.step()

        if epoch % args.eval_step == 0:
            model.eval()
            out = model(g.x, g.edge_index)
            out = F.log_softmax(out, dim=-1)
            train_acc = evaluate(out, 'train')
            valid_acc = evaluate(out, 'valid')
            test_acc = evaluate(out, 'test')
            if valid_acc > best_val:
                best_val = valid_acc
                best_test = test_acc
            print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, '
                  f'Train: {100 * train_acc:.2f}%, '
                  f'Valid: {100 * valid_acc:.2f}%, '
                  f'Test: {100 * test_acc:.2f}%')
    print(f'Best Test Acc: {100 * best_test:.2f}%')


if __name__ == '__main__':
    # acc = 73.44%, 73.39%
    parser = argparse.ArgumentParser(description='Polynormer Example')
    # dataset and evaluation
    parser.add_argument('--dataset', type=str, default='ogbn-arxiv')
    parser.add_argument('--data_dir', type=str, default='./data/ogb')
    parser.add_argument('--local_epochs', type=int, default=2000,
                        help='warmup epochs for local attention')
    parser.add_argument('--global_epochs', type=int, default=500,
                        help='epochs for local-to-global attention')

    # model
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--local_layers', type=int, default=7,
                        help='number of layers for local attention')
    parser.add_argument('--global_layers', type=int, default=2,
                        help='number of layers for global attention')
    parser.add_argument('--num_heads', type=int, default=1,
                        help='number of heads for attention')
    parser.add_argument('--beta', type=float, default=0.9,
                        help='Polynormer beta initialization')
    parser.add_argument('--pre_ln', action='store_true')
    parser.add_argument('--post_bn', action='store_true')
    parser.add_argument('--local_attn', action='store_true')

    # training
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--in_dropout', type=float, default=0.15)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--global_dropout', type=float, default=0.5)

    # display
    parser.add_argument('--eval_step', type=int, default=9,
                        help='how often to evaluate')
    args = parser.parse_args()
    print(args)
    seed_everything(42)
    main(args)
