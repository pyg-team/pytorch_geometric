import argparse
import os
import time
from typing import Optional

import cupy
import rmm
import torch
from rmm.allocators.cupy import rmm_cupy_allocator
from rmm.allocators.torch import rmm_torch_allocator

# Must change allocators immediately upon import
# or else other imports will cause memory to be
# allocated and prevent changing the allocator
rmm.reinitialize(devices=[0], pool_allocator=True, managed_memory=True)
cupy.cuda.set_allocator(rmm_cupy_allocator)
torch.cuda.memory.change_current_allocator(rmm_torch_allocator)

import cugraph  # noqa
import torch.nn.functional as F  # noqa
from cugraph.testing.mg_utils import enable_spilling  # noqa
from cugraph_pyg.data import CuGraphStore  # noqa
from cugraph_pyg.loader import CuGraphNeighborLoader  # noqa

import torch_geometric  # noqa
from torch_geometric.loader import NeighborLoader  # noqa

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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

enable_spilling()

from ogb.nodeproppred import PygNodePropPredDataset  # noqa

dataset = PygNodePropPredDataset(name='ogbn-papers100M',
                                 root='/datasets/ogb_datasets')
split_idx = dataset.get_idx_split()
data = dataset[0]

G = {("N", "E", "N"): data.edge_index}
N = {"N": data.num_nodes}
fs = cugraph.gnn.FeatureStore(backend="torch")
fs.add_data(data.x, "N", "x")
fs.add_data(data.y, "N", "y")
cugraph_store = CuGraphStore(fs, G, N)
train_loader = CuGraphNeighborLoader(cugraph_store,
                                     input_nodes=split_idx['train'],
                                     shuffle=True, drop_last=True, **kwargs)
val_loader = CuGraphNeighborLoader(cugraph_store,
                                   input_nodes=split_idx['valid'], **kwargs)
test_loader = CuGraphNeighborLoader(cugraph_store,
                                    input_nodes=split_idx['test'], **kwargs)

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
        batch = batch.to_homogeneous()

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
        batch = batch.to_homogeneous()
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
