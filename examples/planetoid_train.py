import argparse
import os.path as osp
import sys
import time

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

import torch_geometric.transforms as T
from torch_geometric import seed_everything
from torch_geometric.datasets import Planetoid
from torch_geometric.logging import init_wandb, log
from torch_geometric.nn import (
    PMLP,
    AGNNConv,
    ARMAConv,
    BatchNorm,
    DNAConv,
    GATConv,
    GCN2Conv,
    GCNConv,
    GraphUNet,
    Linear,
    MixHopConv,
    Node2Vec,
    SGConv,
    SplineConv,
    SuperGATConv,
    TAGConv,
)
from torch_geometric.typing import (
    WITH_TORCH_SPLINE_CONV,
    Adj,
    OptTensor,
    SparseTensor,
)
from torch_geometric.utils import dropout_edge

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter, )
parser.add_argument(
    '--dataset',
    type=str,
    default='Cora',
    choices=['Cora', 'CiteSeer', 'PubMed'],
    help='Dataset name.',
)
parser.add_argument(
    '--dataset_dir',
    type=str,
    default='./data',
    help='Root directory of dataset.',
)
parser.add_argument(
    '--gnn_choice',
    type=str,
    default='arma',
    choices=[
        'agnn', 'arma', 'mixhop', 'sgc', 'tagcn', 'graphunet', 'pmlp',
        'splinegnn', 'gcn2', 'super_gat', 'dna', 'gat', 'gcn', 'node2vec'
    ],
    help='Model name.',
)
parser.add_argument('-e', '--epochs', type=int, default=200)
parser.add_argument('--num_val', type=int, default=500)
parser.add_argument('--num_test', type=int, default=500)
parser.add_argument('--hidden_channels', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--wd', type=float, default=5e-5)
parser.add_argument('--use_normalize_features', action='store_true',
                    help='Use NormalizeFeatures')
parser.add_argument('--use_target_indegree', action='store_true',
                    help='Use TargetIndegree')
parser.add_argument('--use_gdc', action='store_true', help='Use GDC')
parser.add_argument('--wandb', action='store_true', help='Track experiment')
parser.add_argument('--default_split', action='store_true',
                    help='Use default split')
args = parser.parse_args()

if args.gnn_choice == 'splinegnn' and not WITH_TORCH_SPLINE_CONV:
    quit("This example requires 'torch-spline-conv'")

wall_clock_start = time.perf_counter()
seed_everything(123)

if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

# init wandb
if args.wandb:
    init_wandb(name=f'{args.gnn_choice}-{args.dataset}', epochs=args.epochs,
               hidden_channels=args.hidden_channels, lr=args.lr, device=device)

# define dataset, default train=140, val=500, test=1000
print(f'Training {args.dataset} with {args.gnn_choice} model.')

transform_lst = []

if not args.default_split:
    transform_lst.append(
        T.RandomNodeSplit(num_val=args.num_val, num_test=args.num_test))

if args.use_normalize_features:
    transform_lst.append(T.NormalizeFeatures())

if args.use_target_indegree:
    transform_lst.append(T.TargetIndegree())
elif args.use_gdc:
    transform_lst.append(
        T.GDC(
            self_loop_weight=1,
            normalization_in='sym',
            normalization_out='col',
            diffusion_kwargs=dict(method='ppr', alpha=0.05),
            sparsification_kwargs=dict(method='topk', k=128, dim=0),
            exact=True,
        ))

if args.gnn_choice == 'gcn2':
    transform_lst.extend([T.GCNNorm(), T.ToSparseTensor()])

root = osp.join(args.dataset_dir, args.dataset)
print(f'The root is: {root}')
dataset = Planetoid(root, args.dataset, transform=T.Compose(transform_lst))
data = dataset[0].to(device)


# define models
class AGNN(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int,
                 out_channels: int) -> None:
        super().__init__()
        self.lin1 = torch.nn.Linear(in_channels, hidden_channels)
        self.prop1 = AGNNConv(requires_grad=False)
        self.prop2 = AGNNConv(requires_grad=True)
        self.lin2 = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        x = F.dropout(x, training=self.training)
        x = F.relu(self.lin1(x))
        x = self.prop1(x, edge_index)
        x = self.prop2(x, edge_index)
        x = F.dropout(x, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=1)


class ARMA(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int,
                 out_channels: int) -> None:
        super().__init__()
        self.conv1 = ARMAConv(in_channels, hidden_channels, num_stacks=3,
                              num_layers=2, shared_weights=True, dropout=0.25)
        self.conv2 = ARMAConv(hidden_channels, out_channels, num_stacks=3,
                              num_layers=2, shared_weights=True, dropout=0.25,
                              act=lambda x: x)

    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        x = F.dropout(x, training=self.training)
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


class MixHop(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int,
                 out_channels: int) -> None:
        super().__init__()
        self.conv1 = MixHopConv(in_channels, hidden_channels, powers=[0, 1, 2])
        self.norm1 = BatchNorm(3 * hidden_channels)
        self.conv2 = MixHopConv(3 * hidden_channels, hidden_channels,
                                powers=[0, 1, 2])
        self.norm2 = BatchNorm(3 * hidden_channels)
        self.conv3 = MixHopConv(3 * hidden_channels, hidden_channels,
                                powers=[0, 1, 2])
        self.norm3 = BatchNorm(3 * hidden_channels)
        self.lin = Linear(3 * hidden_channels, out_channels)

    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        x = F.dropout(x, p=0.7, training=self.training)
        x = self.conv1(x, edge_index)
        x = self.norm1(x)
        x = F.dropout(x, p=0.9, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.norm2(x)
        x = F.dropout(x, p=0.9, training=self.training)
        x = self.conv3(x, edge_index)
        x = self.norm3(x)
        x = F.dropout(x, p=0.9, training=self.training)
        return F.log_softmax(self.lin(x), dim=1)


class SGC(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = SGConv(in_channels, out_channels, K=2, cached=True)

    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        x = self.conv1(x, edge_index)
        return F.log_softmax(x, dim=1)


class TAGCN(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int,
                 out_channels: int) -> None:
        super().__init__()
        self.conv1 = TAGConv(in_channels, hidden_channels)
        self.conv2 = TAGConv(hidden_channels, out_channels)

    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


class GraphUNetModel(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int,
                 out_channels: int, num_nodes: int) -> None:
        super().__init__()
        pool_ratios = [2000 / num_nodes, 0.5]
        self.unet = GraphUNet(in_channels, hidden_channels, out_channels,
                              depth=3, pool_ratios=pool_ratios)

    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        edge_index, _ = dropout_edge(edge_index, p=0.2, force_undirected=True,
                                     training=self.training)
        x = F.dropout(x, p=0.92, training=self.training)

        x = self.unet(x, edge_index)
        return F.log_softmax(x, dim=1)


class SplineGNN(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int,
                 out_channels: int) -> None:
        super().__init__()
        self.conv1 = SplineConv(in_channels, hidden_channels, dim=1,
                                kernel_size=2)
        self.conv2 = SplineConv(hidden_channels, out_channels, dim=1,
                                kernel_size=2)

    def forward(self, x: Tensor, edge_index: Adj, edge_attr: OptTensor = None):
        x = F.dropout(x, training=self.training)
        x = F.elu(self.conv1(x, edge_index, edge_attr))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        return F.log_softmax(x, dim=1)


class GCN2(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int,
                 out_channels: int, num_layers: int, alpha: float,
                 theta: float, shared_weights: bool = True,
                 dropout: float = 0.0) -> None:
        super().__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(Linear(in_channels, hidden_channels))
        self.lins.append(Linear(hidden_channels, out_channels))

        self.convs = torch.nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(
                GCN2Conv(hidden_channels, alpha, theta, layer + 1,
                         shared_weights, normalize=False))

        self.dropout = dropout

    def forward(self, x: Tensor, adj_t: SparseTensor) -> Tensor:
        x = F.dropout(x, self.dropout, training=self.training)
        x = x_0 = self.lins[0](x).relu()

        for conv in self.convs:
            x = F.dropout(x, self.dropout, training=self.training)
            x = conv(x, x_0, adj_t)
            x = x.relu()

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.lins[1](x)

        return x.log_softmax(dim=-1)


class SuperGAT(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        self.conv1 = SuperGATConv(in_channels, 8, heads=8, dropout=0.6,
                                  attention_type='MX', edge_sample_ratio=0.8,
                                  is_undirected=True)
        self.conv2 = SuperGATConv(8 * 8, out_channels, heads=8, concat=False,
                                  dropout=0.6, attention_type='MX',
                                  edge_sample_ratio=0.8, is_undirected=True)

    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        att_loss = self.conv1.get_attention_loss()
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, data.edge_index)
        att_loss += self.conv2.get_attention_loss()
        return F.log_softmax(x, dim=-1), att_loss


class DNA(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int,
                 out_channels: int, num_layers: int, heads: int = 1,
                 groups: int = 1) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.lin1 = torch.nn.Linear(in_channels, hidden_channels)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(
                DNAConv(hidden_channels, heads, groups, dropout=0.8))
        self.lin2 = torch.nn.Linear(hidden_channels, out_channels)

    def reset_parameters(self) -> None:
        self.lin1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x_all = x.view(-1, 1, self.hidden_channels)
        for conv in self.convs:
            x = F.relu(conv(x_all, edge_index))
            x = x.view(-1, 1, self.hidden_channels)
            x_all = torch.cat([x_all, x], dim=1)
        x = x_all[:, -1]
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return torch.log_softmax(x, dim=1)


class GAT(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int,
                 out_channels: int, heads: int) -> None:
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads, dropout=0.6)
        # On the Pubmed dataset, use `heads` output heads in `conv2`.
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1,
                             concat=False, dropout=0.6)

    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


class GCN(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int,
                 out_channels: int) -> None:
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels,
                             normalize=not args.use_gdc)
        self.conv2 = GCNConv(hidden_channels, out_channels,
                             normalize=not args.use_gdc)

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv1(x, edge_index, edge_weight).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)


def get_model(gnn_choice: str) -> torch.nn.Module:
    if gnn_choice == 'agnn':
        model = AGNN(
            dataset.num_features,
            args.hidden_channels,
            dataset.num_classes,
        )
    elif gnn_choice == 'arma':
        model = ARMA(
            dataset.num_features,
            args.hidden_channels,
            dataset.num_classes,
        )
    elif gnn_choice == 'mixhop':
        model = MixHop(
            dataset.num_features,
            args.hidden_channels,
            dataset.num_classes,
        )
    elif gnn_choice == 'sgc':
        model = SGC(
            dataset.num_features,
            dataset.num_classes,
        )
    elif gnn_choice == 'tagcn':
        model = TAGCN(
            dataset.num_features,
            args.hidden_channels,
            dataset.num_classes,
        )
    elif gnn_choice == 'graphunet':
        model = GraphUNetModel(
            dataset.num_features,
            args.hidden_channels,
            dataset.num_classes,
            data.num_nodes,
        )
    elif gnn_choice == 'pmlp':
        model = PMLP(
            in_channels=dataset.num_features,
            hidden_channels=args.hidden_channels,
            out_channels=dataset.num_classes,
            num_layers=2,
            dropout=0.5,
            norm=False,
        )
    elif gnn_choice == 'splinegnn':
        model = SplineGNN(
            dataset.num_features,
            args.hidden_channels,
            dataset.num_classes,
        )
    elif gnn_choice == 'gcn2':
        model = GCN2(
            dataset.num_features,
            args.hidden_channels,
            dataset.num_classes,
            num_layers=64,
            alpha=0.1,
            theta=0.5,
            shared_weights=True,
            dropout=0.6,
        )
    elif gnn_choice == 'super_gat':
        model = SuperGAT(
            dataset.num_features,
            dataset.num_classes,
        )
    elif gnn_choice == 'dna':
        model = DNA(
            dataset.num_features,
            args.hidden_channels,
            dataset.num_classes,
            num_layers=5,
            heads=8,
            groups=16,
        )
    elif gnn_choice == 'gat':
        model = GAT(
            dataset.num_features,
            args.hidden_channels,
            dataset.num_classes,
            heads=8,
        )
    elif gnn_choice == 'gcn':
        model = GCN(
            dataset.num_features,
            args.hidden_channels,
            dataset.num_classes,
        )
    elif gnn_choice == 'node2vec':
        model = Node2Vec(
            data.edge_index,
            embedding_dim=args.hidden_channels,
            walk_length=20,
            context_size=10,
            walks_per_node=10,
            num_negative_samples=1,
            p=1.0,
            q=1.0,
            sparse=True,
        )
    else:
        raise ValueError(f'Unsupported model type: {gnn_choice}')
    return model


def get_optimizer(
    gnn_choice: str,
    model: torch.nn.Module,
) -> torch.optim.Optimizer:
    if gnn_choice == 'node2vec':
        optimizer = torch.optim.SparseAdam(model.parameters(), lr=args.lr)
    elif gnn_choice == 'gcn':
        optimizer = torch.optim.Adam([
            dict(params=model.conv1.parameters(), weight_decay=args.wd),
            dict(params=model.conv2.parameters(), weight_decay=0),
        ], lr=args.lr)
    elif gnn_choice == 'mixhop':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                    weight_decay=args.wd)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                     weight_decay=args.wd)
    return optimizer


model = get_model(args.gnn_choice).to(device)
optimizer = get_optimizer(args.gnn_choice, model)
if args.gnn_choice == 'node2vec':
    num_workers = 4 if sys.platform == 'linux' else 0
    loader = model.loader(batch_size=128, shuffle=True,
                          num_workers=num_workers)


def train_node2vec():
    model.train()
    total_loss = torch.zeros(1, device=device).squeeze()
    for pos_rw, neg_rw in loader:
        optimizer.zero_grad()
        loss = model.loss(pos_rw.to(device), neg_rw.to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss
    return total_loss.item() / len(loader)


@torch.no_grad()
def test_node2vec():
    model.eval()
    z = model()
    accs = []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        acc = model.test(
            train_z=z[data.train_mask],
            train_y=data.y[data.train_mask],
            test_z=z[mask],
            test_y=data.y[mask],
            max_iter=150,
        )
        accs.append(acc)
    return accs


def train():
    model.train()
    optimizer.zero_grad()
    # forward
    if args.gnn_choice in ['splinegnn', 'gcn']:
        out = model(data.x, data.edge_index, data.edge_attr)
    elif args.gnn_choice == 'gcn2':
        out = model(data.x, data.adj_t)
    elif args.gnn_choice == 'super_gat':
        out, att_loss = model(data.x, data.edge_index)
    else:
        out = model(data.x, data.edge_index)
    # loss
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    if args.gnn_choice == 'super_gat':
        loss += 4.0 * att_loss
    # backward
    loss.backward()
    optimizer.step()
    return loss


@torch.no_grad()
def test():
    model.eval()
    accs = []
    # forward
    if args.gnn_choice in ['splinegnn', 'gcn']:
        out = model(data.x, data.edge_index, data.edge_attr)
    elif args.gnn_choice == 'gcn2':
        out = model(data.x, data.adj_t)
    elif args.gnn_choice == 'super_gat':
        out, _ = model(data.x, data.edge_index)
    else:
        out = model(data.x, data.edge_index)
    # accuracy
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = out[mask].argmax(1)
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


print(f'Total time before training begins took '
      f'{time.perf_counter() - wall_clock_start:.4f}s')
print('Training...')

# add tensorboard
writer = SummaryWriter()

times = []
best_val_acc = test_acc = 0.
for epoch in range(args.epochs):
    start = time.perf_counter()
    if args.gnn_choice == 'node2vec':
        train_loss = train_node2vec()
        train_acc, val_acc, tmp_test_acc = test_node2vec()
    else:
        train_loss = train()
        train_acc, val_acc, tmp_test_acc = test()
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc
    log(Epoch=epoch, Loss=f'{train_loss:.4f}', Train=f'{train_acc:.4f}',
        Val=f'{best_val_acc:.4f}', Test=f'{test_acc:.4f}')
    times.append(time.perf_counter() - start)

    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Accuracy/train', train_acc, epoch)
    writer.add_scalar('Accuracy/val', val_acc, epoch)
    writer.add_scalar('Accuracy/test', test_acc, epoch)

print(f'Average Epoch Time: {torch.tensor(times).mean():.4f}s')
print(f'Median Epoch Time: {torch.tensor(times).median():.4f}s')
print(f'Best Validation Accuracy: {100.0 * best_val_acc:.2f}%')
print(f'Test Accuracy: {100.0 * test_acc:.2f}%')
print(f'Total Program Runtime: '
      f'{time.perf_counter() - wall_clock_start:.4f}s')


@torch.no_grad()
def plot_points(colors):
    model.eval()
    z = model().cpu().numpy()
    z = TSNE(n_components=2).fit_transform(z)
    y = data.y.cpu().numpy()

    plt.figure(figsize=(8, 8))
    for i in range(dataset.num_classes):
        plt.scatter(z[y == i, 0], z[y == i, 1], s=20, color=colors[i])
    plt.axis('off')
    plt.show()


if args.gnn_choice == 'node2vec':
    colors = [
        '#ffc0cb', '#bada55', '#008080', '#420420', '#7fe5f0', '#065535',
        '#ffd700'
    ]
    plot_points(colors)
