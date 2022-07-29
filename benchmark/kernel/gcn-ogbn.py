import argparse
import time

import torch
import torch.nn.functional as F
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
from torch.profiler import ProfilerActivity, profile

import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.profile import rename_profile_file, trace_handler


class Net(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, use_sage) -> None:
        super(Net, self).__init__()
        if use_sage:
            conv_layer_0 = SAGEConv(in_channels, hidden_channels)
            conv_layer_1 = SAGEConv(hidden_channels, hidden_channels)
            conv_layer_2 = SAGEConv(hidden_channels, out_channels)
        else:
            conv_layer_0 = GCNConv(in_channels, hidden_channels,
                                   normalize=False)
            conv_layer_1 = GCNConv(hidden_channels, hidden_channels,
                                   normalize=False)
            conv_layer_2 = GCNConv(hidden_channels, out_channels,
                                   normalize=False)

        self.convs = torch.nn.ModuleList()
        self.convs.append(conv_layer_0)
        for _ in range(num_layers - 2):
            self.convs.append(conv_layer_1)
        self.convs.append(conv_layer_2)

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return torch.log_softmax(x, dim=-1)


def train(model, data, train_idx, optimizer):
    model.train()

    optimizer.zero_grad()
    out = model(data.x, data.adj_t)[train_idx]
    loss = F.nll_loss(out, data.y.squeeze(1)[train_idx])
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test(model, data, split_idx, evaluator):
    model.eval()

    out = model(data.x, data.adj_t)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': data.y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': data.y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': data.y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, valid_acc, test_acc


@torch.no_grad()
def inference(model, data):
    model.eval()
    model(data.x, data.adj_t)


def main():
    parser = argparse.ArgumentParser(description='OGBN-Products (GNN)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--use_sage', action='store_true')
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--inference', action='store_true')
    parser.add_argument('--profile', action='store_true')
    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = PygNodePropPredDataset(name='ogbn-products', root='dataset/',
                                     transform=T.ToSparseTensor())
    data = dataset[0]

    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train'].to(device)

    model = Net(data.num_features, args.hidden_channels, dataset.num_classes,
                args.num_layers, args.dropout, args.use_sage).to(device)

    if not args.use_sage:
        # Pre-compute GCN normalization.
        adj_t = data.adj_t.set_diag()
        deg = adj_t.sum(dim=1).to(torch.float)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
        data.adj_t = adj_t

    data = data.to(device)

    if not args.inference:
        evaluator = Evaluator(name='ogbn-products')

        for run in range(args.runs):
            model.reset_parameters()
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            for epoch in range(1, 1 + args.epochs):
                loss = train(model, data, train_idx, optimizer)
                result = test(model, data, split_idx, evaluator)

                if epoch % args.log_steps == 0:
                    train_acc, valid_acc, test_acc = result
                    print(f'Run: {run + 1:02d}, '
                          f'Epoch: {epoch:02d}, '
                          f'Loss: {loss:.4f}, '
                          f'Train: {100 * train_acc:.2f}%, '
                          f'Valid: {100 * valid_acc:.2f}% '
                          f'Test: {100 * test_acc:.2f}%')
    else:
        model.reset_parameters()
        for epoch in range(1, 1 + args.epochs):
            print("Epoch ", epoch)
            if epoch == args.epochs:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_start = time.time()

            inference(model, data)

            if epoch == args.epochs:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_end = time.time()
                duration = t_end - t_start
                print(f'End-to-End Inference time: {duration:.8f}s',
                      flush=True)

        if args.profile:
            with profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    on_trace_ready=trace_handler) as p:
                inference(model, data)
                p.step()
            rename_profile_file('GCN' if not args.use_sage else 'SAGE',
                                'ogbn-products')


if __name__ == "__main__":
    main()
