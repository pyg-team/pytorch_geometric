import argparse

import torch
import torch.nn.functional as F
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset

from torch_geometric import seed_everything
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.nn.attention import PolynormerAttention
from torch_geometric.utils import (
    add_self_loops,
    remove_self_loops,
    to_undirected,
)


class Net(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        local_layers=3,
        global_layers=2,
        in_dropout=0.15,
        dropout=0.5,
        global_dropout=0.5,
        heads=1,
        beta=0.9,
        pre_ln=False,
        post_bn=True,
        local_attn=False,
    ):
        super().__init__()

        self._global = False
        self.in_drop = in_dropout
        self.dropout = dropout
        self.pre_ln = pre_ln
        self.post_bn = post_bn

        self.beta = beta

        self.h_lins = torch.nn.ModuleList()
        self.local_convs = torch.nn.ModuleList()
        self.lins = torch.nn.ModuleList()
        self.lns = torch.nn.ModuleList()
        if self.pre_ln:
            self.pre_lns = torch.nn.ModuleList()
        if self.post_bn:
            self.post_bns = torch.nn.ModuleList()

        # first layer
        inner_channels = heads * hidden_channels
        self.h_lins.append(torch.nn.Linear(in_channels, inner_channels))
        if local_attn:
            self.local_convs.append(
                GATConv(in_channels, hidden_channels, heads=heads, concat=True,
                        add_self_loops=False, bias=False))
        else:
            self.local_convs.append(
                GCNConv(in_channels, inner_channels, cached=False,
                        normalize=True))

        self.lins.append(torch.nn.Linear(in_channels, inner_channels))
        self.lns.append(torch.nn.LayerNorm(inner_channels))
        if self.pre_ln:
            self.pre_lns.append(torch.nn.LayerNorm(in_channels))
        if self.post_bn:
            self.post_bns.append(torch.nn.BatchNorm1d(inner_channels))

        # following layers
        for _ in range(local_layers - 1):
            self.h_lins.append(torch.nn.Linear(inner_channels, inner_channels))
            if local_attn:
                self.local_convs.append(
                    GATConv(inner_channels, hidden_channels, heads=heads,
                            concat=True, add_self_loops=False, bias=False))
            else:
                self.local_convs.append(
                    GCNConv(inner_channels, inner_channels, cached=False,
                            normalize=True))

            self.lins.append(torch.nn.Linear(inner_channels, inner_channels))
            self.lns.append(torch.nn.LayerNorm(inner_channels))
            if self.pre_ln:
                self.pre_lns.append(torch.nn.LayerNorm(heads *
                                                       hidden_channels))
            if self.post_bn:
                self.post_bns.append(torch.nn.BatchNorm1d(inner_channels))

        self.lin_in = torch.nn.Linear(in_channels, inner_channels)
        self.ln = torch.nn.LayerNorm(inner_channels)
        self.global_attn = PolynormerAttention(
            hidden_channels,
            heads,
            global_layers,
            beta,
            global_dropout,
        )
        self.pred_local = torch.nn.Linear(inner_channels, out_channels)
        self.pred_global = torch.nn.Linear(inner_channels, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        for local_conv in self.local_convs:
            local_conv.reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()
        for h_lin in self.h_lins:
            h_lin.reset_parameters()
        for ln in self.lns:
            ln.reset_parameters()
        if self.pre_ln:
            for p_ln in self.pre_lns:
                p_ln.reset_parameters()
        if self.post_bn:
            for p_bn in self.post_bns:
                p_bn.reset_parameters()
        self.lin_in.reset_parameters()
        self.ln.reset_parameters()
        self.global_attn.reset_parameters()
        self.pred_local.reset_parameters()
        self.pred_global.reset_parameters()

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.in_drop, training=self.training)

        # equivariant local attention
        x_local = 0
        for i, local_conv in enumerate(self.local_convs):
            if self.pre_ln:
                x = self.pre_lns[i](x)
            h = self.h_lins[i](x)
            h = F.relu(h)
            x = local_conv(x, edge_index) + self.lins[i](x)
            if self.post_bn:
                x = self.post_bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = (1 - self.beta) * self.lns[i](h * x) + self.beta * x
            x_local = x_local + x

        # equivariant global attention
        if self._global:
            x_global = self.global_attn(self.ln(x_local))
            x = self.pred_global(x_global)
        else:
            x = self.pred_local(x_local)

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
