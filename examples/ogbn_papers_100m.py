import argparse
import os
import time
from typing import Optional

import torch
import torch.nn.functional as F
from ogb.nodeproppred import PygNodePropPredDataset

import torch_geometric
from torch_geometric.loader import NeighborLoader

parser = argparse.ArgumentParser()
parser.add_argument('--hidden_channels', type=int, default=256)
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--fan_out', type=int, default=30)
parser.add_argument(
    "--use_gat_conv",
    action='store_true',
    help="Wether or not to use GATConv. (Defaults to using GCNConv)",
)
parser.add_argument(
    "--n_gat_conv_heads",
    type=int,
    default=4,
    help="If using GATConv, number of attention heads to use",
)
args = parser.parse_args()
wall_clock_start = time.perf_counter()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = PygNodePropPredDataset(name='ogbn-papers100M',
                                 root='/datasets/ogb_datasets')
split_idx = dataset.get_idx_split()


def get_num_workers() -> int:
    try:
        return len(os.sched_getaffinity(0)) // 2
    except Exception:
        return os.cpu_count() // 2


kwargs = dict(
    num_neighbors=[args.fan_out] * args.num_layers,
    batch_size=args.batch_size,
)
# Set Up Neighbor Loading
data = dataset[0]
num_work = get_num_workers()
train_loader = NeighborLoader(data=data, input_nodes=split_idx['train'],
                              num_workers=num_work, drop_last=True,
                              shuffle=False, **kwargs)
val_loader = NeighborLoader(data=data, input_nodes=split_idx['valid'],
                            num_workers=num_work, **kwargs)
test_loader = NeighborLoader(data=data, input_nodes=split_idx['test'],
                             num_workers=num_work, **kwargs)

if args.use_gat_conv:
    model = torch_geometric.nn.models.GAT(
        dataset.num_features, args.hidden_channels, args.num_layers,
        dataset.num_classes, heads=args.n_gat_conv_heads).to(device)
else:
    model = torch_geometric.nn.models.GCN(
        dataset.num_features,
        args.hidden_channels,
        args.num_layers,
        dataset.num_classes,
    ).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                             weight_decay=0.0005)

warmup_steps = 20


def train():
    model.train()
    for i, batch in enumerate(train_loader):
        if i == warmup_steps:
            torch.cuda.synchronize()
            start_avg_time = time.perf_counter()
        batch = batch.to(device)
        optimizer.zero_grad()
        batch_size = batch.num_sampled_nodes[0]
        out = model(batch.x, batch.edge_index)[:batch_size]
        y = batch.y[:batch_size].view(-1).to(torch.long)
        loss = F.cross_entropy(out, y)
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print(f'Epoch: {epoch:02d}, Iteration: {i}, Loss: {loss:.4f}')
    torch.cuda.synchronize()
    print(f'Average Training Iteration Time (s/iter): \
            {(time.perf_counter() - start_avg_time)/(i-warmup_steps):.6f}')


@torch.no_grad()
def test(loader: NeighborLoader, val_steps: Optional[int] = None):
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


torch.cuda.synchronize()
prep_time = round(time.perf_counter() - wall_clock_start, 2)
print("Total time before training begins (prep_time)=", prep_time, "seconds")
print("Beginning training...")
for epoch in range(1, 1 + args.epochs):
    train()
    val_acc = test(val_loader, val_steps=100)
    print(f'Val Acc: ~{val_acc:.4f}')

test_acc = test(test_loader)
print(f'Test Acc: {test_acc:.4f}')
total_time = round(time.perf_counter() - wall_clock_start, 2)
print("Total Program Runtime (total_time) =", total_time, "seconds")
print("total_time - prep_time =", total_time - prep_time, "seconds")
