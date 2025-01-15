import argparse
import os.path as osp
import time

import psutil
import torch
import torch.nn.functional as F
from ogb.nodeproppred import PygNodePropPredDataset
from torch import Tensor
from tqdm import tqdm

from torch_geometric.loader import NeighborLoader
from torch_geometric.nn.models import GAT, GraphSAGE
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
    '--use_gat',
    action='store_true',
    help='Whether or not to use GAT model',
)
parser.add_argument('-e', '--epochs', type=int, default=10)
parser.add_argument('--num_layers', type=int, default=3)
parser.add_argument('--num_heads', type=int, default=2,
                    help='number of heads for GAT model.')
parser.add_argument('-b', '--batch_size', type=int, default=1024)
parser.add_argument('--num_workers', type=int, default=12)
parser.add_argument('--fan_out', type=int, default=10,
                    help='number of neighbors in each layer')
parser.add_argument('--hidden_channels', type=int, default=256)
parser.add_argument('--lr', type=float, default=0.003)
parser.add_argument('--wd', type=float, default=0.0)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument(
    '--use_directed_graph',
    action='store_true',
    help='Whether or not to use directed graph',
)
args = parser.parse_args()

wall_clock_start = time.perf_counter()

if (args.dataset == 'ogbn-papers100M'
        and (psutil.virtual_memory().total / (1024**3)) < 390):
    print('Warning: may not have enough RAM to run this example.')
    print('Consider upgrading RAM if an error occurs.')
    print('Estimated RAM Needed: ~390GB.')

if args.use_gat:
    print(f'Training {args.dataset} with GAT model.')
else:
    print(f'Training {args.dataset} with GraphSage model.')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_epochs = args.epochs
num_layers = args.num_layers
num_workers = args.num_workers
num_hidden_channels = args.hidden_channels
batch_size = args.batch_size
root = osp.join(args.dataset_dir, args.dataset)
print('The root is: ', root)
dataset = PygNodePropPredDataset(name=args.dataset, root=root)
split_idx = dataset.get_idx_split()
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


def train(epoch: int) -> tuple[Tensor, float]:
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
def test(loader: NeighborLoader) -> float:
    model.eval()

    total_correct = total_examples = 0
    for batch in loader:
        batch = batch.to(device)
        batch_size = batch.num_sampled_nodes[0]
        out = model(batch.x, batch.edge_index)[:batch_size]
        pred = out.argmax(dim=-1)
        y = batch.y[:batch_size].view(-1).to(torch.long)

        total_correct += int((pred == y).sum())
        total_examples += y.size(0)

    return total_correct / total_examples


if args.use_gat:
    model = GAT(
        in_channels=dataset.num_features,
        hidden_channels=num_hidden_channels,
        num_layers=num_layers,
        out_channels=dataset.num_classes,
        dropout=args.dropout,
        heads=args.num_heads,
    )
else:
    model = GraphSAGE(
        in_channels=dataset.num_features,
        hidden_channels=num_hidden_channels,
        num_layers=num_layers,
        out_channels=dataset.num_classes,
        dropout=args.dropout,
    )

model = model.to(device)
model.reset_parameters()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=args.lr,
    weight_decay=args.wd,
)

print(f'Total time before training begins took '
      f'{time.perf_counter() - wall_clock_start:.4f}s')
print('Training...')

times = []
train_times = []
inference_times = []
best_val = 0.
for epoch in range(1, num_epochs + 1):
    train_start = time.perf_counter()
    loss, _ = train(epoch)
    train_times.append(time.perf_counter() - train_start)

    inference_start = time.perf_counter()
    val_acc = test(val_loader)
    inference_times.append(time.perf_counter() - inference_start)

    print(f'Epoch {epoch:02d}, Loss: {loss:.4f}, ',
          f'Train Time: {train_times[-1]:.4f}s')
    print(f'Val: {val_acc * 100.0:.2f}%,')

    if val_acc > best_val:
        best_val = val_acc
    times.append(time.perf_counter() - train_start)

print(f'Average Epoch Time on training: '
      f'{torch.tensor(train_times).mean():.4f}s')
print(f'Average Epoch Time on inference: '
      f'{torch.tensor(inference_times).mean():.4f}s')
print(f'Average Epoch Time: {torch.tensor(times).mean():.4f}s')
print(f'Median Epoch Time: {torch.tensor(times).median():.4f}s')
print(f'Best Validation Accuracy: {100.0 * best_val:.2f}%')

print('Testing...')
test_final_acc = test(test_loader)
print(f'Test Accuracy: {100.0 * test_final_acc:.2f}%')
print(f'Total Program Runtime: '
      f'{time.perf_counter() - wall_clock_start:.4f}s')
