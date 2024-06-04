# Reaches around 0.7870 ± 0.0036 test accuracy.

import argparse
import os.path as osp
import time

import torch
import torch.nn.functional as F
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
from tqdm import tqdm

from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import to_undirected

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument("-e", "--epochs", type=int, default=10,
                    help='number of training epochs.')
parser.add_argument("--runs", type=int, default=4, help='number of runs.')
parser.add_argument("--num_layers", type=int, default=3,
                    help='number of layers.')
parser.add_argument("-b", "--batch_size", type=int, default=1024,
                    help='batch size.')
parser.add_argument("--workers", type=int, default=12,
                    help='number of workers.')
parser.add_argument("--neighbors", type=str, default='15,10,5',
                    help='number of neighbors.')
parser.add_argument("--fan_out", type=int, default=10,
                    help='number of fanout.')
parser.add_argument("--log_interval", type=int, default=10,
                    help='number of log interval.')
parser.add_argument("--hidden_channels", type=int, default=256,
                    help='number of hidden channels.')
parser.add_argument("--lr", type=float, default=0.003)
parser.add_argument("--wd", type=float, default=0.00)
parser.add_argument("--dropout", type=float, default=0.5)
parser.add_argument(
    "--use_undirected_graph",
    default = "True",
    choices=["False", "True"],
    help="Wether or not to use undirected graph",
)
args = parser.parse_args()
wall_clock_start = time.perf_counter()
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device(args.device)
num_epochs = args.epochs
num_runs = args.runs
num_layers = args.num_layers
num_workers = args.workers
num_hidden_channels = args.hidden_channels
batch_size = args.batch_size
neighbors = args.neighbors.split(',')
num_neighbors = [int(i) for i in neighbors]
log_interval = args.log_interval

root = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data',
                'ogb-papers100M')
print("The root is: ", root)
dataset = PygNodePropPredDataset('ogbn-papers100M', root)
split_idx = dataset.get_idx_split()
evaluator = Evaluator(name='ogbn-papers100M')
data = dataset[0]
use_undirected_graph = args.use_undirected_graph == "True":
if use_undirected_graph:
    start_undirected = time.time()
    print("use undirected graph")
    data.edge_index = to_undirected(data.edge_index, reduce="mean")
    print("to_undirected is done")
    end_undirected = time.time()
    total_undirected_time = round(end_undirected - start_undirected, 2)
    print("to_undirected costs ", total_undirected_time, "second")
data.to(device, 'x', 'y')

train_loader = NeighborLoader(
    data,
    input_nodes=split_idx['train'],
    num_neighbors=[args.fan_out] * num_layers,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    persistent_workers=True,
)
val_loader = NeighborLoader(
    data,
    input_nodes=split_idx['valid'],
    num_neighbors=[args.fan_out] * num_layers,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    persistent_workers=True,
)
test_loader = NeighborLoader(
    data,
    input_nodes=split_idx['test'],
    num_neighbors=[args.fan_out] * num_layers,
    #num_neighbors=[2*args.fan_out] * num_layers,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    persistent_workers=True,
)
subgraph_loader = NeighborLoader(
    data,
    input_nodes=None,
    num_neighbors=[-1],
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=True,
)


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super().__init__()

        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != self.num_layers - 1:
                x = x.relu()
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def inference(self, x_all):
        pbar = tqdm(total=x_all.size(0) * self.num_layers)
        pbar.set_description('Evaluating')

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
        for i in range(self.num_layers):
            xs = []
            for batch in subgraph_loader:
                x = x_all[batch.n_id].to(device)
                edge_index = batch.edge_index.to(device)
                x = self.convs[i](x, edge_index)
                x = x[:batch.batch_size]
                if i != self.num_layers - 1:
                    x = x.relu()
                xs.append(x.cpu())

                pbar.update(batch.batch_size)

            x_all = torch.cat(xs, dim=0)

        pbar.close()

        return x_all


model = SAGE(dataset.num_features, num_hidden_channels, dataset.num_classes,
             num_layers=num_layers, dropout=args.dropout)
model = model.to(device)


def train(epoch):
    model.train()

    pbar = tqdm(total=split_idx['train'].size(0))
    pbar.set_description(f'Epoch {epoch:02d}')

    total_loss = total_correct = 0
    for batch in train_loader:
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index.to(device))[:batch.batch_size]
        y = batch.y[:batch.batch_size].squeeze().to(torch.long)
        loss = F.cross_entropy(out, y)
        loss.backward()
        optimizer.step()

        total_loss += float(loss)
        total_correct += int(out.argmax(dim=-1).eq(y).sum())
        pbar.update(batch.batch_size)

    pbar.close()

    loss = total_loss / len(train_loader)
    approx_acc = total_correct / split_idx['train'].size(0)

    return loss, approx_acc


@torch.no_grad()
def test(loader: NeighborLoader, val_steps=None):
    model.eval()

    total_correct = total_examples = 0
    for i, batch in enumerate(loader):
        if val_steps is not None and i >= val_steps:
            break
        batch = batch.to(device)
        batch_size = batch.num_sampled_nodes[0]
        out = model(batch.x, batch.edge_index)[:batch_size]
        pred = out.argmax(dim=-1)
        y = batch.y[:batch_size].view(-1).to(torch.long)

        total_correct += int((pred == y).sum())
        total_examples += y.size(0)

    return total_correct / total_examples


test_accs = []
val_accs = []
#for run in range(1, 11):
times = []
train_times = []
inference_times = []
best_val = best_test = 0.
prep_time = round(time.perf_counter() - wall_clock_start, 2)
print("Total time before training begins (prep_time)=", prep_time, "seconds")
print("Beginning training...")
for run in range(1, num_runs + 1):
    start = time.time()
    print(f'\nRun {run:02d}:\n')

    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = best_test_acc = 0.0
    for epoch in range(1, num_epochs + 1):
        train_start = time.time()
        loss, acc = train(epoch)
        train_end = time.time()
        train_times.append(train_end - train_start)
        print(f'Epoch {epoch:02d}, Loss: {loss:.4f}, Approx. Train: {acc:.4f}'
              f' Time: {train_end - train_start:.4f}s')

        inference_start = time.time()
        train_acc = test(test_loader)
        val_acc = test(val_loader)
        test_acc = test(test_loader)
        inference_times.append(time.time() - inference_start)
        test_accs.append(test_acc)
        val_accs.append(val_acc)
        #if log_interval and epoch % log_interval:
        print(f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, '
              f'Test: {test_acc:.4f}')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
        if test_acc > best_test_acc:
            best_test_acc = test_acc
        times.append(time.time() - train_start)
    if best_val < best_val_acc:
        best_val = best_val_acc
    if best_test < best_test_acc:
        best_test = best_test_acc
    print("Total time used for run: {:02d} is {:.4f}".format(
        run,
        time.time() - start))

test_acc = torch.tensor(test_accs)
val_acc = torch.tensor(val_accs)
print('============================')
print("Average Epoch Time on training: {:.4f}".format(
    torch.tensor(train_times).mean()))
print("Average Epoch Time on inference: {:.4f}".format(
    torch.tensor(inference_times).mean()))
print("Average Epoch Time: {:.4f}".format(torch.tensor(times).mean()))
print(f"Median time per epoch: {torch.tensor(times).median():.4f}s")
print(f'Final Test: {test_acc.mean():.4f} ± {test_acc.std():.4f}')
print(f'Final Validation: {val_acc.mean():.4f} ± {val_acc.std():.4f}')
print(f"Best validation accuracy: {best_val:.4f}")
print(f"Best testing accuracy: {best_test:.4f}")
test_final_acc = test(test_loader)
print(f'Test Accuracy: {test_final_acc:.4f}')
total_time = round(time.perf_counter() - wall_clock_start, 2)
print("Total Program Runtime (total_time) =", total_time, "seconds")
