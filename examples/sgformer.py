"""This example run SGFormer model on ogbn-arxiv dataset.
Original Paper: https://arxiv.org/abs/2306.10759
"SGFormer: Simplifying and Empowering Transformers for
Large-Graph Representations".
"""
import argparse

import torch
import torch.nn.functional as F
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset

from torch_geometric import seed_everything
from torch_geometric.nn.attention import SGFormerAttention
from torch_geometric.nn.conv import GCNConv
from torch_geometric.utils import (
    add_self_loops,
    remove_self_loops,
    to_undirected,
)


class GraphModule(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        num_layers=2,
        dropout=0.5,
    ):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        self.fcs = torch.nn.ModuleList()
        self.fcs.append(torch.nn.Linear(in_channels, hidden_channels))

        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))

        self.dropout = dropout
        self.activation = F.relu

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        for fc in self.fcs:
            fc.reset_parameters()

    def forward(self, x, edge_index):
        x = self.fcs[0](x)
        x = self.bns[0](x)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        last_x = x

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.bns[i + 1](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + last_x
        return x


class SGModule(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        num_layers=2,
        num_heads=1,
        dropout=0.5,
    ):
        super().__init__()

        self.attns = torch.nn.ModuleList()
        self.fcs = torch.nn.ModuleList()
        self.fcs.append(torch.nn.Linear(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.LayerNorm(hidden_channels))
        for _ in range(num_layers):
            self.attns.append(
                SGFormerAttention(hidden_channels, num_heads, hidden_channels))
            self.bns.append(torch.nn.LayerNorm(hidden_channels))

        self.dropout = dropout
        self.activation = F.relu

    def reset_parameters(self):
        for attn in self.attns:
            attn.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        for fc in self.fcs:
            fc.reset_parameters()

    def forward(self, x):
        layer_ = []

        # input MLP layer
        x = self.fcs[0](x)
        x = self.bns[0](x)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # store as residual link
        layer_.append(x)

        for i, attn in enumerate(self.attns):
            x = attn(x)
            x = (x + layer_[i]) / 2.
            x = self.bns[i + 1](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            layer_.append(x)

        return x


class Net(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        trans_num_layers=1,
        trans_num_heads=1,
        trans_dropout=0.5,
        gnn_num_layers=1,
        gnn_dropout=0.5,
        graph_weight=0.8,
        aggregate='add',
    ):
        super().__init__()
        self.trans_conv = SGModule(
            in_channels,
            hidden_channels,
            trans_num_layers,
            trans_num_heads,
            trans_dropout,
        )
        self.graph_conv = GraphModule(
            in_channels,
            hidden_channels,
            gnn_num_layers,
            gnn_dropout,
        )
        self.graph_weight = graph_weight

        self.aggregate = aggregate

        if aggregate == 'add':
            self.fc = torch.torch.nn.Linear(hidden_channels, out_channels)
        elif aggregate == 'cat':
            self.fc = torch.torch.nn.Linear(2 * hidden_channels, out_channels)
        else:
            raise ValueError(f'Invalid aggregate type:{aggregate}')

        self.params1 = list(self.trans_conv.parameters())
        self.params2 = list(self.graph_conv.parameters())
        self.params2.extend(list(self.fc.parameters()))

    def forward(self, x, edge_index):
        x1 = self.trans_conv(x)
        x2 = self.graph_conv(x, edge_index)
        if self.aggregate == 'add':
            x = self.graph_weight * x2 + (1 - self.graph_weight) * x1
        else:
            x = torch.cat((x1, x2), dim=1)
        x = self.fc(x)
        return x


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
        graph_weight=args.graph_weight,
        aggregate=args.aggregate,
        trans_num_layers=args.trans_num_layers,
        trans_dropout=args.trans_dropout,
        trans_num_heads=args.trans_num_heads,
        gnn_num_layers=args.gnn_num_layers,
        gnn_dropout=args.gnn_dropout,
    ).to(device)
    criterion = torch.torch.nn.NLLLoss()
    optimizer = torch.optim.Adam([
        {
            'params': model.params1,
            'weight_decay': args.trans_weight_decay
        },
        {
            'params': model.params2,
            'weight_decay': args.gnn_weight_decay
        },
    ], lr=args.lr)

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
    for epoch in range(args.epochs):
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
    parser = argparse.ArgumentParser(description='SGFormer Example')
    # dataset and evaluation
    parser.add_argument('--dataset', type=str, default='ogbn-arxiv')
    parser.add_argument('--data_dir', type=str, default='./data/ogb')
    parser.add_argument('--epochs', type=int, default=1000)

    # gnn branch
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--aggregate', type=str, default='add',
                        help='aggregate type, add or cat.')
    parser.add_argument('--graph_weight', type=float, default=0.5,
                        help='graph weight.')

    parser.add_argument('--gnn_num_layers', type=int, default=3,
                        help='number of layers for GNN')
    parser.add_argument('--gnn_dropout', type=float, default=0.5)
    parser.add_argument('--gnn_weight_decay', type=float, default=0.)

    # all-pair attention (Transformer) branch
    parser.add_argument('--trans_num_heads', type=int, default=1,
                        help='number of heads for attention')
    parser.add_argument('--trans_num_layers', type=int, default=2,
                        help='number of layers for all-pair attention.')
    parser.add_argument('--trans_dropout', type=float, default=0.5)
    parser.add_argument('--trans_weight_decay', type=float, default=0.)

    # training
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=10000,
                        help='mini batch training for large graphs')

    # display and utility
    parser.add_argument('--eval_step', type=int, default=9,
                        help='how often to evaluate')
    args = parser.parse_args()
    print(args)
    seed_everything(123)
    main(args)
