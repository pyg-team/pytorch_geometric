# Reaches around 0.7870 ± 0.0036 test accuracy.

import argparse
import os.path as osp

import torch
import torch.nn.functional as F
from ogb.nodeproppred import PygNodePropPredDataset
from tqdm import tqdm

from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import to_undirected

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--num_layers', type=int, default=3)
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--num_neighbors', type=int, default=10)
parser.add_argument('--channels', type=int, default=256)
parser.add_argument('--lr', type=float, default=0.003)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--num_workers', type=int, default=12)
args = parser.parse_args()

root = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'papers100')
dataset = PygNodePropPredDataset('ogbn-papers100M', root)
split_idx = dataset.get_idx_split()

data = dataset[0]
data.edge_index = to_undirected(data.edge_index, reduce="mean")

train_loader = NeighborLoader(
    data,
    input_nodes=split_idx['train'],
    num_neighbors=[args.num_neighbors] * args.num_layers,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
    persistent_workers=args.num_workers > 0,
)
val_loader = NeighborLoader(
    data,
    input_nodes=split_idx['valid'],
    num_neighbors=[args.num_neighbors] * args.num_layers,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    persistent_workers=args.num_workers > 0,
)
test_loader = NeighborLoader(
    data,
    input_nodes=split_idx['test'],
    num_neighbors=[args.num_neighbors] * args.num_layers,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    persistent_workers=args.num_workers > 0,
)


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, args.channels))
        for _ in range(args.num_layers - 2):
            self.convs.append(SAGEConv(args.channels, args.channels))
        self.convs.append(SAGEConv(args.channels, out_channels))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != args.num_layers - 1:
                x = x.relu()
                x = F.dropout(x, p=args.dropout, training=self.training)
        return x


model = SAGE(dataset.num_features, dataset.num_classes).to(args.device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


def train():
    model.train()

    total_loss = total_correct = total_examples = 0
    for batch in tqdm(train_loader):
        batch = batch.to(args.device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index)[:batch.batch_size]
        y = batch.y[:batch.batch_size].view(-1).to(torch.long)
        loss = F.cross_entropy(out, y)
        loss.backward()
        optimizer.step()

        total_loss += float(loss) * y.size(0)
        total_correct += int(out.argmax(dim=-1).eq(y).sum())
        total_examples += y.size(0)

    return total_loss / total_examples, total_correct / total_examples


@torch.no_grad()
def test(loader):
    model.eval()

    total_correct = total_examples = 0
    for batch in tqdm(loader):
        batch = batch.to(args.device)
        out = model(batch.x, batch.edge_index)[:batch.batch_size]
        y = batch.y[:batch.batch_size].view(-1).to(torch.long)

        total_correct += int(out.argmax(dim=-1).eq(y).sum())
        total_examples += y.size(0)

    return total_correct / total_examples


for epoch in range(1, args.epochs + 1):
    loss, train_acc = train()
    print(f'Epoch {epoch:02d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}')
    val_acc = test(val_loader)
    print(f'Epoch {epoch:02d}, Val Acc: {val_acc:.4f}')
    test_acc = test(test_loader)
    print(f'Epoch {epoch:02d}, Test Acc: {test_acc:.4f}')
