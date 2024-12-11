import argparse
import os
import os.path as osp
import tempfile

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

import cugraph_pyg  # noqa
import torch.nn.functional as F  # noqa
# Enable cudf spilling to save gpu memory
from cugraph.testing.mg_utils import enable_spilling  # noqa
from cugraph_pyg.loader import NeighborLoader  # noqa

enable_spilling()

from ogb.nodeproppred import PygNodePropPredDataset  # noqa
from tqdm import tqdm  # noqa

from torch_geometric.nn import SAGEConv  # noqa
from torch_geometric.utils import to_undirected  # noqa

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--num_layers', type=int, default=3)
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--num_neighbors', type=int, default=10)
parser.add_argument('--channels', type=int, default=256)
parser.add_argument('--lr', type=float, default=0.003)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--num_workers', type=int, default=12)
parser.add_argument('--root', type=str, default=None)
parser.add_argument('--tempdir_root', type=str, default=None)
args = parser.parse_args()

root = args.root
if root is None:
    root = osp.dirname(osp.realpath(__file__))
    root = osp.join(root, '..', 'data', 'papers100')

dataset = PygNodePropPredDataset('ogbn-papers100M', root)
split_idx = dataset.get_idx_split()

data = dataset[0]
data.edge_index = to_undirected(data.edge_index, reduce="mean")

graph_store = cugraph_pyg.data.GraphStore()
graph_store[dict(
    edge_type=('node', 'rel', 'node'),
    layout='coo',
    is_sorted=False,
    size=(data.num_nodes, data.num_nodes),
)] = data.edge_index

feature_store = cugraph_pyg.data.TensorDictFeatureStore()
feature_store['node', 'x', None] = data.x
feature_store['node', 'y', None] = data.y

data = (feature_store, graph_store)


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


model = SAGE(dataset.num_features, dataset.num_classes).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


def create_loader(
    data,
    num_neighbors,
    input_nodes,
    replace,
    batch_size,
    samples_dir,
    stage_name,
    shuffle=False,
):
    directory = osp.join(samples_dir, stage_name)
    os.mkdir(directory)
    return NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        input_nodes=input_nodes,
        replace=replace,
        batch_size=batch_size,
        directory=directory,
        shuffle=shuffle,
    )


with tempfile.TemporaryDirectory(dir=args.tempdir_root) as samples_dir:
    loader_kwargs = dict(
        data=data,
        num_neighbors=[args.num_neighbors] * args.num_layers,
        replace=False,
        batch_size=args.batch_size,
        samples_dir=samples_dir,
    )

    train_loader = create_loader(
        input_nodes=split_idx['train'],
        stage_name='train',
        shuffle=True,
        **loader_kwargs,
    )

    val_loader = create_loader(
        input_nodes=split_idx['valid'],
        stage_name='val',
        **loader_kwargs,
    )

    test_loader = create_loader(
        input_nodes=split_idx['test'],
        stage_name='test',
        **loader_kwargs,
    )

    def train():
        model.train()

        total_loss = total_correct = total_examples = 0
        for i, batch in enumerate(train_loader):
            batch = batch.cuda()
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)[:batch.batch_size]
            y = batch.y[:batch.batch_size].view(-1).to(torch.long)
            loss = F.cross_entropy(out, y)
            loss.backward()
            optimizer.step()

            total_loss += float(loss) * y.size(0)
            total_correct += int(out.argmax(dim=-1).eq(y).sum())
            total_examples += y.size(0)

            if i % 10 == 0:
                print(f"Epoch: {epoch:02d}, Iteration: {i}, Loss: {loss:.4f}")

        return total_loss / total_examples, total_correct / total_examples

    @torch.no_grad()
    def test(loader):
        model.eval()

        total_correct = total_examples = 0
        for batch in loader:
            batch = batch.cuda()
            out = model(batch.x, batch.edge_index)[:batch.batch_size]
            y = batch.y[:batch.batch_size].view(-1).to(torch.long)

            total_correct += int(out.argmax(dim=-1).eq(y).sum())
            total_examples += y.size(0)

        return total_correct / total_examples

    for epoch in range(1, args.epochs + 1):
        loss, train_acc = train()
        print(f'Epoch {epoch:02d}, Loss: {loss:.4f}, Train: {train_acc:.4f}')
        val_acc = test(val_loader)
        print(f'Epoch {epoch:02d}, Val: {val_acc:.4f}')
        test_acc = test(test_loader)
        print(f'Epoch {epoch:02d}, Test: {test_acc:.4f}')
