import argparse
import os.path as osp
import time
from typing import Tuple

import psutil
import torch
import torch.nn.functional as F
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
from torch import Tensor
from tqdm import tqdm

from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import to_undirected

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter, )
parser.add_argument(
    '--dataset',
    type=str,
    default='ogbn-papers100M',
    choices=['ogbn-papers100M', 'ogbn-products'],
    help='Dataset name.',
)
parser.add_argument(
    '--dataset_dir',
    type=str,
    default='./data',
    help='Root directory of dataset.',
)
parser.add_argument(
    "--dataset_subdir",
    type=str,
    default="ogb-papers100M",
    help="directory of dataset.",
)
parser.add_argument(
    '--use_gat',
    action='store_true',
    help='Whether or not to use GAT model',
)
parser.add_argument(
    '--verbose',
    action='store_true',
    help='Whether or not to generate statistical report',
)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('-e', '--epochs', type=int, default=10,
                    help='number of training epochs.')
parser.add_argument('--num_layers', type=int, default=3,
                    help='number of layers.')
parser.add_argument('--num_heads', type=int, default=2,
                    help='number of heads for GAT model.')
parser.add_argument('-b', '--batch_size', type=int, default=1024,
                    help='batch size.')
parser.add_argument('--num_workers', type=int, default=12,
                    help='number of workers.')
parser.add_argument('--fan_out', type=int, default=10,
                    help='number of neighbors in each layer')
parser.add_argument('--hidden_channels', type=int, default=256,
                    help='number of hidden channels.')
parser.add_argument('--lr', type=float, default=0.003)
parser.add_argument('--wd', type=float, default=0.0,
                    help='weight decay for the optimizer')
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument(
    '--use_directed_graph',
    action='store_true',
    help='Whether or not to use directed graph',
)
args = parser.parse_args()
if "papers" in str(args.dataset) and (psutil.virtual_memory().total /
                                      (1024**3)) < 390:
    print("Warning: may not have enough RAM to use this many GPUs.")
    print("Consider upgrading RAM if an error occurs.")
    print("Estimated RAM Needed: ~390GB.")
verbose = args.verbose
if verbose:
    wall_clock_start = time.perf_counter()
    if args.use_gat:
        print(f"Training {args.dataset} with GAT model.")
    else:
        print(f"Training {args.dataset} with GraphSage model.")

if not torch.cuda.is_available():
    args.device = "cpu"
device = torch.device(args.device)

num_epochs = args.epochs
num_layers = args.num_layers
num_workers = args.num_workers
num_hidden_channels = args.hidden_channels
batch_size = args.batch_size

root = osp.join(args.dataset_dir, args.dataset_subdir)
print('The root is: ', root)
dataset = PygNodePropPredDataset(name=args.dataset, root=root)
split_idx = dataset.get_idx_split()
evaluator = Evaluator(name=args.dataset)

data = dataset[0]
if not args.use_directed_graph:
    data.edge_index = to_undirected(data.edge_index, reduce='mean')

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
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    persistent_workers=True,
)


def train(epoch: int) -> Tuple[Tensor, float]:
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
def test(loader: NeighborLoader, val_steps=None) -> float:
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


if args.use_gat:
    from torch_geometric.nn.models import GAT
    model = GAT(
        in_channels=dataset.num_features,
        hidden_channels=num_hidden_channels,
        num_layers=num_layers,
        out_channels=dataset.num_classes,
        dropout=args.dropout,
        heads=args.num_heads,
    )
else:
    from torch_geometric.nn.models import GraphSAGE
    model = GraphSAGE(
        in_channels=dataset.num_features,
        hidden_channels=num_hidden_channels,
        num_layers=num_layers,
        out_channels=dataset.num_classes,
        dropout=args.dropout,
    )

model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                             weight_decay=args.wd)

if verbose:
    prep_time = round(time.perf_counter() - wall_clock_start, 2)
    print("Total time before training begins (prep_time)=", prep_time,
          "seconds")
    print("Training...")

test_accs = []
val_accs = []
times = []
train_times = []
inference_times = []
best_val = best_test = 0.
start = time.time()

model.reset_parameters()

for epoch in range(1, num_epochs + 1):
    train_start = time.time()
    loss, _ = train(epoch)
    train_end = time.time()
    train_times.append(train_end - train_start)

    inference_start = time.time()
    val_acc = test(val_loader)
    test_acc = test(test_loader)

    inference_times.append(time.time() - inference_start)
    test_accs.append(test_acc)
    val_accs.append(val_acc)
    print(
        f'Epoch {epoch:02d}, Loss: {loss:.4f}, Train Time: {train_end - train_start:.4f}s'
    )
    print(f'Val: {val_acc * 100.0:.2f}%,', f'Test: {test_acc * 100.0:.2f}%')

    if val_acc > best_val:
        best_val = val_acc
    if test_acc > best_test:
        best_test = test_acc
    times.append(time.time() - train_start)

if verbose:
    test_acc = torch.tensor(test_accs)
    val_acc = torch.tensor(val_accs)
    print('============================')
    print("Average Epoch Time on training: {:.4f}".format(
        torch.tensor(train_times).mean()))
    print("Average Epoch Time on inference: {:.4f}".format(
        torch.tensor(inference_times).mean()))
    print(f"Average Epoch Time: {torch.tensor(times).mean():.4f}")
    print(f"Median time per epoch: {torch.tensor(times).median():.4f}s")
    print(f'Final Test: {test_acc.mean():.4f} ± {test_acc.std():.4f}')
    print(f'Final Validation: {val_acc.mean():.4f} ± {val_acc.std():.4f}')
    print(f"Best validation accuracy: {best_val:.4f}")
    print(f"Best testing accuracy: {best_test:.4f}")

if verbose:
    print("Testing...")
test_final_acc = test(test_loader)
print(f'Test Accuracy: {100.0 * test_final_acc:.2f}%')
if verbose:
    total_time = round(time.perf_counter() - wall_clock_start, 2)
    print("Total Program Runtime (total_time) =", total_time, "seconds")
